"""
自定义奖励函数模块 - 为Libero任务提供密集奖励信号
"""

import numpy as np
from gym import spaces
import wandb
# 存储所有任务类型的奖励函数的字典
reward_functions = {}

# 任务ID到奖励函数的映射
task_id_to_reward = {
    # libero_object 任务集
    "pick_up_the_milk_and_place_it_in_the_basket": "pick_and_place_in_basket",
    "pick_up_the_butter_and_place_it_in_the_basket": "pick_and_place_in_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket": "pick_and_place_in_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket": "pick_and_place_in_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket": "pick_and_place_in_basket",
}

def get_reward_function_by_task(task_id=None, task_name=None, task_description=None):
    """
    根据任务ID、名称或描述获取适当的奖励函数
    
    Args:
        task_id: 任务ID（BDDL文件名）
        task_name: 任务名称
        task_description: 任务描述
        
    Returns:
        reward_function: 选择的奖励函数
    """
    # 尝试通过任务ID匹配
    if task_id is not None and task_id in task_id_to_reward:
        reward_function_name = task_id_to_reward[task_id]
        if reward_function_name in reward_functions:
            print(f"[info] 通过任务ID匹配到奖励函数: {reward_function_name}")
            return reward_functions[reward_function_name]
    
    # 尝试通过任务名称或描述匹配关键词
    for source in [task_name, task_description]:
        if source is not None:
            source_lower = source.lower()
            
            # 检查是否为"拾取并放入篮子"类任务
            if (("pick" in source_lower and "place" in source_lower and "basket" in source_lower) or
                ("pick up" in source_lower and "in" in source_lower and "basket" in source_lower)):
                print(f"[info] 通过关键词匹配到奖励函数: pick_and_place_in_basket")
                return reward_functions['pick_and_place_in_basket']
    
    # 默认返回基础奖励函数
    print(f"[info] 未找到匹配的奖励函数，使用默认奖励函数")
    return reward_functions['default']

# ====================== 辅助函数 ======================

def get_goal_state(env):
    """
    安全地从环境中获取目标状态，适用于不同环境结构
    """
    # 尝试不同的方式获取目标状态
    try:
        # 直接从环境获取
        if hasattr(env, 'parsed_problem'):
            return env.parsed_problem["goal_state"]
        # 从嵌套环境获取
        elif hasattr(env, 'env') and hasattr(env.env, 'parsed_problem'):
            return env.env.parsed_problem["goal_state"]
        # 尝试从info中获取
        elif hasattr(env, 'previous_state') and 'goal_state' in env.previous_state:
            return env.previous_state['goal_state']
        else:
            # 如果无法获取，返回空列表作为默认值
            print("[警告] 无法从环境中获取目标状态")
            return []
    except Exception as e:
        print(f"[错误] 获取目标状态时出错: {str(e)}")
        return []

def get_distance(pos1, pos2):
    """计算两点之间的欧几里得距离"""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def is_grasped(gripper_state):
    """检查物体是否被抓取（基于抓手状态）"""
    return gripper_state < 0.035  # 抓手关闭时值通常小于-0.5

def is_lifted(obj_pos, table_height=0.0):
    """检查物体是否被抬起（基于物体高度和桌面高度）"""
    # print('obj_pos hieght:',obj_pos[2])
    return obj_pos[2] > 0.12  # 高于桌面5cm表示被抬起

def get_bounding_box_size(obj_info):
    """获取物体包围盒大小"""
    if 'bounding_box' in obj_info:
        bbox = obj_info['bounding_box']
        return np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]])
    return np.array([0.1, 0.1, 0.1])  # 默认大小

def is_above(pos1, pos2, threshold=0.1):
    """检查pos1是否在pos2上方"""
    return (abs(pos1[0] - pos2[0]) < threshold and
            abs(pos1[1] - pos2[1]) < threshold and
            pos1[2] > pos2[2])

# ====================== 通用奖励函数 ======================

def default_reward(env, state):
    """
    默认奖励函数：奖励接近目标对象和距离改进
    
    Args:
        env: 环境对象
        state: 当前状态
        
    Returns:
        reward: 计算的奖励值
        info: 额外信息
    """
    # 默认奖励为0
    reward = 0.0
    info = {'reward_stage': 'exploration'}

    # 获取基本信息
    if 'robot_state' not in state or 'object_states' not in state:
        return reward, info

    # 获取机器人状态
    robot_pos = state['robot_state']['gripper_position']
    
    # 如果有目标对象，奖励接近它
    if 'target_object' in state and state['target_object'] in state['object_states']:
        target_obj = state['target_object']
        target_pos = state['object_states'][target_obj]['position']
        
        # 计算距离
        distance = get_distance(robot_pos, target_pos)
        reward -= 0.1 * distance  # 距离越小奖励越大
        
        info['reward_stage'] = 'approaching_target'
        
        # 如果之前的状态存在，奖励距离改进
        if hasattr(env, 'previous_state') and 'object_states' in env.previous_state:
            prev_target_pos = env.previous_state['object_states'][target_obj]['position']
            prev_distance = get_distance(robot_pos, prev_target_pos)
            
            # 距离改进奖励
            distance_improvement = prev_distance - distance
            reward += 0.5 * distance_improvement
    
    return reward, info

# ====================== 任务特定奖励函数 ======================
def pick_and_place_in_basket_reward(env, state):
    """
    拾取并放入篮子的奖励函数
    
    Args:
        env: 环境对象
        state: 当前状态
        
    Returns:
        reward: 计算的奖励值
        info: 额外信息
    """
    reward = 0.0
    info = {'reward_stage': 'exploration'}
    log_rewards = getattr(env, 'log_rewards', False)
    
    # 检查状态有效性
    if 'robot_state' not in state or 'object_states' not in state:
        return reward, info
    
    # 获取机器人状态
    gripper_pos = state['robot_state']['gripper_position']
    gripper_state = state['robot_state']['gripper']  # 抓手开合状态
    
    # 获取目标对象和篮子
    target_obj = None
    basket_obj = None
    
    # 查找任务中的目标对象和篮子
    if 'target_object' in state:
        target_obj_base = state['target_object']
        # 查找完整的目标对象名称（通常以基础名称开头）
        for obj_name in state['object_states']:
            if obj_name.startswith(target_obj_base) and '_main' in obj_name:
                target_obj = obj_name
                break
        # 如果没有找到带_main的，就用基础名称
        if target_obj is None:
            target_obj = target_obj_base
    
    # 查找篮子（通常名称中包含'basket'）
    for obj_name in state['object_states']:
        if 'basket' in obj_name.lower():
            basket_obj = obj_name
            break
    
    # 如果没有找到目标对象或篮子，使用默认奖励
    if target_obj is None or basket_obj is None or target_obj not in state['object_states'] or basket_obj not in state['object_states']:
        if log_rewards:
            print(f"[info] Target object or basket not found: target={target_obj}, basket={basket_obj}")
        return default_reward(env, state)
    
    # 获取目标对象和篮子的位置
    target_pos = state['object_states'][target_obj]['position']
    basket_pos = state['object_states'][basket_obj]['position']
    
    # 获取桌子高度（如果可用）
    table_height = 0.0
    try:
        # 尝试从环境中获取桌子高度
        if 'table' in state['object_states']:
            table_height = state['object_states']['table']['position'][2]
        # 尝试从其他物体推断
        elif len(state['object_states']) > 0:
            heights = [obj['position'][2] for obj in state['object_states'].values() 
                      if 'position' in obj and len(obj['position']) > 2]
            if heights:
                table_height = min(heights)
    except Exception as e:
        print(f"[警告] 获取桌面高度时出错: {str(e)}")
    
    # 初始化距离参考值（如果还没有）
    if not hasattr(env, 'reward_stage'):
        env.reward_stage = 0
    if not hasattr(env, 'initial_distance_to_target'):
        env.initial_distance_to_target = get_distance(gripper_pos, target_pos)
    if not hasattr(env, 'initial_distance_to_basket'):
        env.initial_distance_to_basket = get_distance(target_pos, basket_pos)
    
    # 计算当前距离
    distance_to_target = get_distance(gripper_pos, target_pos)
    distance_to_basket = get_distance(target_pos, basket_pos)
    target_height = target_pos[2]
    is_target_lifted = is_lifted(target_pos, table_height)
    is_target_grasped = is_grasped(gripper_state)
    is_above_basket = is_above(target_pos, basket_pos, threshold=0.2)
    
    
    # 根据不同阶段计算奖励
    if not is_target_lifted:
        # 阶段0-2: 接近并抓取物体
        if distance_to_target > 0.5:
            # 阶段0: 远离目标物体
            env.reward_stage = 0
            reward = -2  # 负奖励鼓励接近
            info['reward_stage'] = 'far_from_target'
        elif distance_to_target > 0.2:
            # 阶段1: 接近目标物体
            env.reward_stage = 1
            reward = 2
            info['reward_stage'] = 'approaching_target'
        else:
            # 阶段2: 非常接近目标物体，准备抓取
            env.reward_stage = 2
            reward = 4
            info['reward_stage'] = 'ready_to_grasp'
    elif is_target_lifted and is_target_grasped:
        # 阶段3: 已经抓起物体
        env.reward_stage = 3
        reward = 8
        info['reward_stage'] = 'object_lifted'
        
        # 如果接近篮子，给予额外奖励
        if distance_to_basket < 0.3:
            reward += 4
            info['reward_stage'] = 'approaching_basket'
            
        # 如果在篮子上方，给予更多奖励
        if is_above_basket:
            reward += 4
            info['reward_stage'] = 'above_basket'
    
    # 阶段4: 物体已在篮子中（或非常接近）
    if distance_to_basket < 0.2 and target_height < (basket_pos[2] + 0.15):
        env.reward_stage = 4
        reward = 16
        info['reward_stage'] = 'in_basket'
    
    # 如果物体掉落（曾经达到了阶段3但现在不再被抬起），给予惩罚
    if env.reward_stage >= 3 and not is_target_lifted and not is_target_grasped:
        reward = -4
        info['reward_stage'] = 'object_dropped'
        # 注意，我们不重置reward_stage，因为这样可以保留之前达到的最高阶段
    
    return reward, info
# def pick_and_place_in_basket_reward(env, state):
#     """
#     拾取并放入篮子的奖励函数
    
#     Args:
#         env: 环境对象
#         state: 当前状态
        
#     Returns:
#         reward: 计算的奖励值
#         info: 额外信息
#     """
#     reward = 0.0
#     info = {'reward_stage': 'exploration'}
#     log_rewards = getattr(env, 'log_rewards', False)
    
#     # 检查状态有效性
#     if 'robot_state' not in state or 'object_states' not in state:
#         return reward, info
    
#     # 获取机器人状态
#     gripper_pos = state['robot_state']['gripper_position']
#     gripper_state = state['robot_state']['gripper']  # 抓手开合状态
    
#     # 获取目标对象和篮子
#     target_obj = None
#     basket_obj = None
    
#     # 查找任务中的目标对象和篮子
#     if 'target_object' in state:
#         target_obj_base = state['target_object']
#         # 查找完整的目标对象名称（通常以基础名称开头）
#         for obj_name in state['object_states']:
#             if obj_name.startswith(target_obj_base) and '_main' in obj_name:
#                 target_obj = obj_name
#                 break
#         # 如果没有找到带_main的，就用基础名称
#         if target_obj is None:
#             target_obj = target_obj_base
    
#     # if log_rewards:
#     #     print(f"[info] target_obj: {target_obj}")
    
#     # 查找篮子（通常名称中包含'basket'）
#     for obj_name in state['object_states']:
#         if 'basket' in obj_name.lower():
#             basket_obj = obj_name
#             break
    
#     # 如果没有找到目标对象或篮子，使用默认奖励
#     if target_obj is None or basket_obj is None or target_obj not in state['object_states'] or basket_obj not in state['object_states']:
#         if log_rewards:
#             print(f"[info] Target object or basket not found: target={target_obj}, basket={basket_obj}")
#         return default_reward(env, state)
    
#     # if log_rewards:
#     #     print(f"[info] 找到目标和篮子: target={target_obj}, basket={basket_obj}")
    
#     # 获取目标对象和篮子的位置
#     target_pos = state['object_states'][target_obj]['position']
#     basket_pos = state['object_states'][basket_obj]['position']
    
#     # 获取桌子高度（如果可用）
#     table_height = 0.0
#     try:
#         # 尝试从环境中获取桌子高度
#         if 'table' in state['object_states']:
#             table_height = state['object_states']['table']['position'][2]
#         # 尝试从其他物体推断
#         elif len(state['object_states']) > 0:
#             heights = [obj['position'][2] for obj in state['object_states'].values() 
#                       if 'position' in obj and len(obj['position']) > 2]
#             if heights:
#                 table_height = min(heights)
#     except Exception as e:
#         print(f"[警告] 获取桌面高度时出错: {str(e)}")
    
#     # 定义任务阶段和相应的奖励
#     # 1. 接近目标对象
#     if env.initial_distance_to_target is None:
#         env.initial_distance_to_target = get_distance(gripper_pos, target_pos)
#     if env.initial_distance_to_basket is None:
#         env.initial_distance_to_basket = get_distance(target_pos, basket_pos)
#     distance_to_target = get_distance(gripper_pos, target_pos)
 
#     distance_to_basket = get_distance(target_pos, basket_pos)
#     with open("./distance_log.txt", "a") as f:
#         f.write(f"{distance_to_basket}\n")
#     if distance_to_target < 0.5:
#         env.reward_stage = 1
#         reward = 2 
#         if distance_to_target < 0.2:
#             env.reward_stage = 2
#             reward = 4
#     else:
#         reward = -2  
  
#     if is_lifted(target_pos, table_height):
#         env.reward_stage = 3
#         reward = 8

#     elif env.reward_stage == 3:
#         distance_to_basket = get_distance(target_pos, basket_pos) 
#         if distance_to_basket < 0.2:
#             reward = 16

#     return reward, info

# 注册奖励函数
reward_functions['default'] = default_reward
reward_functions['pick_and_place_in_basket'] = pick_and_place_in_basket_reward 