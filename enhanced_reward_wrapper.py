import gym
import numpy as np
from custom_reward_functions import reward_functions, get_reward_function_by_task

class EnhancedRewardWrapper(gym.Wrapper):
    """
    一个为Libero环境提供增强奖励的包装器，使用自定义奖励函数。
    
    Args:
        env: 要包装的环境
        task_name: 任务名称
        task_id: 任务ID
        task_description: 任务描述
        reward_scale: 奖励缩放系数
        reward_function: 显式指定的奖励函数（可选）
        log_rewards: 是否记录奖励信息（默认False）
    """
    
    def __init__(self, env, task_name=None, task_id=None, task_description=None, 
                 reward_scale=0.1, reward_function=None, log_rewards=False):
        super(EnhancedRewardWrapper, self).__init__(env)
        self.task_name = task_name
        self.task_id = task_id
        self.task_description = task_description
        self.reward_scale = reward_scale
        self.previous_state = None
        self.log_rewards = log_rewards
        self.reward_stage = None
        # 设置奖励函数
        self.initial_distance_to_target = None
        self.initial_distance_to_basket = None
        self.reward_function = reward_function
        if self.reward_function is None:
            # 根据任务信息选择合适的奖励函数
            self.reward_function = self._select_reward_function()
        
        print(f"[info] 使用奖励函数: {self.reward_function.__name__}")
    
    def _select_reward_function(self):
        """基于任务信息选择合适的奖励函数"""
        # 使用新的任务识别函数来选择奖励函数
        return get_reward_function_by_task(
            task_id=self.task_id,
            task_name=self.task_name,
            task_description=self.task_description
        )
    
    def _calculate_dense_reward(self, state, action, next_state, sparse_reward):
        """
        计算密集奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            sparse_reward: 稀疏奖励
            
        Returns:
            reward: 计算的密集奖励
            info: 额外信息
        """
        # 使用选择的奖励函数计算密集奖励
        dense_reward, info = self.reward_function(self, next_state)
        
        # if self.log_rewards:
        #     print(f"[info] dense_reward: {dense_reward}")
            
        # 缩放密集奖励并与稀疏奖励组合
        scaled_dense_reward = dense_reward * self.reward_scale
        
        # 如果获得稀疏奖励，增加总奖励 (稀疏奖励为1表示任务完成)
        if sparse_reward > 0:
            scaled_dense_reward += sparse_reward
            info['reward_stage'] = 'task_complete'
        
        return scaled_dense_reward, info
    
    def step(self, action):
        """
        执行环境步骤，计算并返回增强奖励
        
        Args:
            action: 要执行的动作
            
        Returns:
            observation: 观察
            reward: 增强奖励
            done: 是否完成
            info: 附加信息
        """
        # 保存当前状态
        state = self.previous_state
        
        # 执行环境步骤
        observation, sparse_reward, done, info = self.env.step(action)
        
        # 收集下一个状态信息
        next_state = self._collect_state_info(observation, info)
        
        # 计算增强奖励
        reward, reward_info = self._calculate_dense_reward(state, action, next_state, sparse_reward)
        if done and sparse_reward > 0:
            print(f"[info] 完成任务: {self.task_name}")
            reward = 200
        # 更新previous_state
        self.previous_state = next_state
        
        # 将稀疏奖励和奖励信息添加到info
        info['sparse_reward'] = sparse_reward
        
        if self.log_rewards:
            # 添加奖励组件到info
            info.update(reward_info)
        
        return observation, reward, done, info
    
    def reset(self):
        """
        重置环境并初始化状态跟踪
        
        Returns:
            observation: 初始观察
        """
        observation = self.env.reset()
        
        # 初始化状态
        self.previous_state = self._collect_state_info(observation)
        
        return observation
    
    def _get_gripper_position(self):
        """安全地获取机器人抓手位置"""
        try:
            # 尝试从机器人对象获取抓手位置
            if hasattr(self.env, 'robots') and self.env.robots:
                if hasattr(self.env.robots[0], 'get_ee_position'):
                    # 直接从机器人获取末端执行器位置
                    return self.env.robots[0].get_ee_position()
                elif hasattr(self.env.robots[0], 'gripper'):
                    if hasattr(self.env.robots[0].gripper, 'get_position'):
                        return self.env.robots[0].gripper.get_position()
                    elif hasattr(self.env.robots[0].gripper, 'current_position'):
                        return self.env.robots[0].gripper.current_position
            
            # 尝试从MuJoCo模拟器获取
            if hasattr(self.env, 'sim'):
                # 找到具有"ee"或"gripper"或"panda"的站点
                for site_name in self.env.sim.model.site_names:
                    if any(x in site_name.lower() for x in ['ee', 'gripper', 'panda']):
                        return self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_name)]
            
            # 后备方案：返回默认位置
            return np.array([0.5, 0.0, 0.5])  # 默认在工作空间中心
            
        except Exception as e:
            if self.log_rewards:
                print(f"[警告] 获取抓手位置时出错: {str(e)}")
            return np.array([0.5, 0.0, 0.5])  # 默认在工作空间中心
    
    def _get_gripper_state(self):
        """安全地获取抓手状态（开合程度）"""
        try:
            # 尝试从机器人对象获取抓手状态
            if hasattr(self.env, 'robots') and self.env.robots:
                if hasattr(self.env.robots[0], 'gripper'):
                    if hasattr(self.env.robots[0].gripper, 'get_position_gripper'):
                        return self.env.robots[0].gripper.get_position_gripper()
                    elif hasattr(self.env.robots[0].gripper, 'get_position'):
                        try:
                            return self.env.robots[0].gripper.get_position()[-1]
                        except:
                            pass
                    elif hasattr(self.env.robots[0].gripper, 'current_position'):
                        return self.env.robots[0].gripper.current_position
            
            # 后备方案：从模拟器中获取抓手状态
            if hasattr(self.env, 'sim'):
                try:
                    # 尝试从机器人关节状态提取抓手状态
                    joint_names = self.env.sim.model.joint_names
                    for i, name in enumerate(joint_names):
                        if 'finger' in name.lower() or 'gripper' in name.lower():
                            return self.env.sim.data.qpos[i]
                except:
                    pass
            
            # 后备方案：使用固定值
            return 0.0  # 默认为中性状态
            
        except Exception as e:
            if self.log_rewards:
                print(f"[警告] 获取抓手状态时出错: {str(e)}")
            return 0.0  # 默认为中性状态
    
    def _collect_state_info(self, observation, info=None):
        """
        从观察和info中收集当前状态信息
        
        Args:
            observation: 当前观察
            info: 额外信息（可选）
            
        Returns:
            state: 收集的状态信息字典
        """
        state = {}
        
        # 尝试从环境中获取状态信息
        try:
            # 获取机器人状态（使用安全的访问方法）
            state['robot_state'] = {
                'gripper_position': self._get_gripper_position(),
                'gripper': self._get_gripper_state(),
            }
            
            # 获取物体状态
            state['object_states'] = {}
            
            # 尝试从object_states_dict获取物体位置
            try:
                if hasattr(self.env, 'object_states_dict'):
                    for obj_name, obj_state in self.env.object_states_dict.items():
                        if obj_name not in state['object_states']:
                            state['object_states'][obj_name] = {}
                        
                        if 'pos' in obj_state:
                            state['object_states'][obj_name]['position'] = obj_state['pos']
            except Exception as e:
                if self.log_rewards:
                    print(f"[警告] 从object_states_dict获取对象位置时出错: {str(e)}")
            
            # 如果没有获取到足够的物体信息，尝试从sim中获取
            if not state['object_states'] and hasattr(self.env, 'sim'):
                try:
                    # 遍历所有主体（body）查找物体
                    for i in range(self.env.sim.model.nbody):
                        body_name = self.env.sim.model.body_id2name(i)
                        if body_name and not any(x in body_name.lower() for x in ['robot', 'panda', 'floor', 'table', 'wall']):
                            # 可能是我们感兴趣的对象
                            pos = self.env.sim.data.body_xpos[i]
                            state['object_states'][body_name] = {
                                'position': pos
                            }
                except Exception as e:
                    if self.log_rewards:
                        print(f"[警告] 从sim获取对象位置时出错: {str(e)}")
            
            # 尝试获取目标对象
            try:
                if hasattr(self.env, 'obj_of_interest'):
                    if len(self.env.obj_of_interest) > 0:
                        state['target_object'] = self.env.obj_of_interest[0]
            except Exception as e:
                if self.log_rewards:
                    print(f"[警告] 获取目标对象时出错: {str(e)}")
            
            # 尝试获取目标状态
            try:
                if hasattr(self.env, 'parsed_problem') and 'goal_state' in self.env.parsed_problem:
                    state['goal_state'] = self.env.parsed_problem['goal_state']
            except Exception as e:
                if self.log_rewards:
                    print(f"[警告] 获取目标状态时出错: {str(e)}")
            
            # 如果info包含有用信息，也收集它们
            if info is not None:
                # 从info中提取更多的状态信息
                if 'goal_state' in info:
                    state['goal_state'] = info['goal_state']
                if 'target_object' in info:
                    state['target_object'] = info['target_object']
            
        except Exception as e:
            if self.log_rewards:
                print(f"[警告] 收集状态信息时出错: {str(e)}")
        
        return state

def create_env_with_enhanced_rewards(bddl_file, task_name=None, task_id=None, task_description=None, 
                                   reward_scale=0.5, env_args=None, log_rewards=False):
    """
    创建一个具有增强奖励的Libero环境
    
    Args:
        bddl_file: BDDL文件路径
        task_name: 任务名称（可选）
        task_id: 任务ID（可选）
        task_description: 任务描述（可选）
        reward_scale: 奖励缩放系数（默认0.1）
        env_args: 环境参数（可选）
        log_rewards: 是否记录奖励信息（默认False）
        
    Returns:
        env: 包装好的环境
    """
    from libero.libero.envs import OffScreenRenderEnv
    
    # 设置默认环境参数
    if env_args is None:
        env_args = {}
    
    # 创建基础环境
    env = OffScreenRenderEnv(
        # bddl_file_name=bddl_file,
        **env_args
    )
    
    # 如果没有提供任务名称，但提供了任务描述，使用任务描述作为名称
    if task_name is None and task_description is not None:
        task_name = task_description
    
    # 如果没有提供任务ID，尝试从BDDL文件路径提取
    if task_id is None and bddl_file is not None:
        import os
        bddl_filename = os.path.basename(bddl_file)
        if bddl_filename.endswith('.bddl'):
            task_id = bddl_filename[:-5]  # 去除.bddl后缀
    
    # 包装环境以提供增强奖励
    env = EnhancedRewardWrapper(
        env=env,
        task_name=task_name,
        task_id=task_id,
        task_description=task_description,
        reward_scale=reward_scale,
        log_rewards=log_rewards
    )
    
    return env 