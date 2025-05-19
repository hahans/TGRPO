import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoModelForVision2Seq, AutoProcessor, GenerationConfig
import numpy as np
import copy
import os
import json
from datetime import datetime
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoConfig, AutoImageProcessor,
    AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import copy
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle
)
from openvla_utils import get_processor
from robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
import wandb
from tqdm.auto import tqdm  # 导入tqdm进度条库
# Import enhanced reward wrapper modules
from enhanced_reward_wrapper import EnhancedRewardWrapper, create_env_with_enhanced_rewards
import custom_reward_functions
from peft import LoraConfig, get_peft_model, PeftModel
from t_utils import OpenVLAGRPOConfig, RobotEnvironment, TaskManager

class StepWiseGRPOTrainer:
    """
    步时间GRPO (Group Relative Policy Optimization) 训练器
    
    与传统的GRPO不同，这个实现在每个时间步骤计算组内奖励并应用优化，
    而不是等待完整的轨迹。这样能够更精确地捕捉每个时间步长内不同环境之间的奖励差异。
    
    算法流程:
    1. 对每个环境执行动作并收集当前步的奖励
    2. 在当前步计算所有环境的平均奖励和标准差
    3. 计算每个环境在当前步的优势值(相对于组均值)
    4. 使用这些优势值，结合KL惩罚，计算和优化损失函数
    5. 进入下一个时间步，重复以上步骤
    """
    def __init__(self, model, processor, config: OpenVLAGRPOConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        self.model = model.to(self.device)
        
        # 应用LoRA进行参数高效微调
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules='all-linear',  # 根据模型具体情况修改
            # lora_dropout=0.1,
            # bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        params = [param for param in self.model.parameters() if param.requires_grad]
        
        # 创建冻结的参考模型
        self.ref_model = AutoModelForVision2Seq.from_pretrained(
                "/home/vipuser/hdp/openvla-7b-finetuned-libero-object",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization_config=None
            ).to(self.device)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.processor = processor
        self.config = config
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.base_tokenizer = self.processor.tokenizer
        self.prompt_builder = PurePromptBuilder("openvla")
        # 初始化current_task_name属性
        self.current_task_name = None
        
        # 创建保存指标的目录
        self.metrics_dir = os.path.join(self.config.output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 设置任务和任务管理器
        if self.config.tasks is None:
            # 从任务套件获取所有任务
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[config.task_suite_name]()
            self.config.tasks = [task_suite.get_task(i).name for i in range(len(task_suite.tasks))]
            print(f"[info] Using all tasks from {config.task_suite_name}: {self.config.tasks}")
        
        # 创建任务管理器和环境
        self.task_manager = TaskManager(
            tasks=self.config.tasks,
            num_envs_per_task=self.config.num_environments,
            task_suite_name=self.config.task_suite_name,
            camera_height=self.config.camera_height,
            camera_width=self.config.camera_width,
            image_size=self.config.image_size,
            use_enhanced_rewards=self.config.use_enhanced_rewards,
            reward_scale=self.config.reward_scale,
            log_rewards=self.config.log_rewards
        )
        
        self.optimizer = optim.AdamW(params, lr=self.config.learning_rate)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
        )
        self.batch_size = self.config.batch_size
        
        wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name)
    @classmethod
    def from_task_ids(cls, model, processor, task_suite_name="libero_object", task_ids=None, 
                     output_dir="openvla_grpo_output", **kwargs):
        """
        直接从任务ID创建训练器
        
        Args:
            model: 要微调的OpenVLA模型
            processor: OpenVLA处理器
            task_suite_name: Libero任务套件的名称
            task_ids: 要包含的任务ID列表(如果为None，则包含所有任务)
            output_dir: 保存输出的目录
            **kwargs: OpenVLAGRPOConfig的其他参数
            
        Returns:
            初始化的StepWiseGRPOTrainer
        """
        # 从ID获取任务名称
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        if task_ids is None:
            task_ids = list(range(len(task_suite.tasks)))
            
        task_names = [task_suite.tasks[task_id].name for task_id in task_ids]
        
        # 创建配置
        config = OpenVLAGRPOConfig(
            output_dir=output_dir,
            tasks=task_names,
            task_suite_name=task_suite_name,
            **kwargs
        )
        
        return cls(model=model, processor=processor, config=config)
    def _transform(self,data):
        IGNORE_INDEX=-100
        action_tokenizer=self.action_tokenizer
        action=data['action']
        action_description=action_tokenizer(action)
        tokenizer=self.processor.tokenizer
        instruction=str(data['instruction'])
        lang=f"In: What action should the robot take to {instruction}?\nOut: {action_description} "
        input_ids = tokenizer(lang, add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        input_ids=torch.unsqueeze(input_ids,dim=0)
        labels=torch.unsqueeze(labels,dim=0)
        attention_mask = input_ids.ne(32000)
        return dict(pixel_values=data['pixel_values'], input_ids=input_ids, labels=labels,attention_mask=attention_mask,dataset_name="success")
    
    def _prepare_model_inputs(self, observations, env):
        """
        处理Libero观察以创建OpenVLA模型的输入
        
        Args:
            observations: 原始Libero观察(包含'agentview_image'键)
            env: 环境对象，用于获取任务描述
            
        Returns:
            准备好的模型输入张量字典
        """
        # 从Libero观察中提取图像
        # 获取图像并旋转180度
        # images = observations["agentview_image"]
        images = get_libero_image(observations, 224)
        observation = {
            "full_image": images,
            "state": np.concatenate(
                (observations["robot0_eef_pos"], quat2axisangle(observations["robot0_eef_quat"]), observations["robot0_gripper_qpos"])
            ),
        }
        images = Image.fromarray(observation["full_image"])
        images = images.convert("RGB")
        if isinstance(images, np.ndarray):
            # images = np.rot90(images, k=2, axes=(0, 1))
            os.makedirs(os.path.join(self.config.output_dir, "debug_images"), exist_ok=True)
            timestamp = 1
            image_path = os.path.join(self.config.output_dir, "debug_images", f"image_{timestamp}.png")
            Image.fromarray(images.astype(np.uint8)).save(image_path)
        else:
            # 如果已经是PIL图像，使用PIL的旋转方法
            # images = observations["agentview_image"].rotate(180)
            os.makedirs(os.path.join(self.config.output_dir, "debug_images"), exist_ok=True)
            timestamp = 1
            image_path = os.path.join(self.config.output_dir, "debug_images", f"image_{timestamp}.png")
            images.save(image_path)
        # prompt = f"In: What action should the robot take to {env.task_description.lower()}?\nOut:"
        # inputs = self.processor(prompt, images).to('cuda:0', dtype=torch.bfloat16)
        # print("inputs:",inputs)
        prompt_builder = PurePromptBuilder("openvla")
        
        # 使用任务描述创建文本输入
        texts = env.task_description
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {texts}?")

        # 使用base_tokenizer进行标记化
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        attention_mask = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).attention_mask
        attention_mask = torch.tensor([attention_mask])
        input_ids = torch.tensor([input_ids])
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        # 使用处理器处理输入
        if isinstance(images, np.ndarray):
           images = Image.fromarray(images.astype(np.uint8))
        pixel_values = self.processor.image_processor.apply_transform(images)
        pixel_values = torch.tensor(pixel_values).unsqueeze(0).to(torch.bfloat16)

        inputs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
                
        return inputs,texts
    
    def _compute_sequence_logp(self, vla, inputs):
        """
        计算生成的标记的序列级对数概率
        
        Args:
            model: 用于计算对数概率的模型
            inputs: 包括pixel_values、input_ids等的模型输入
            generated_ids: (B, L) 形状的生成的标记id张量
            num_device: 要使用的设备编号
            
        Returns:
            (B,) 形状的对数概率张量
        """
        # 准备模型输入
        # inputs_ids = pad_sequence(inputs['input_ids'], batch_first=True, padding_value=2)
        # labels = generated_ids
        # labels[:,:-7] = -100
        # labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        # attention_mask = inputs_ids.ne(2)

        
        outputs: CausalLMOutputWithPast = vla(input_ids=inputs["input_ids"].to(self.device),
                        pixel_values=inputs["pixel_values"].to(self.device),
                        attention_mask=inputs["attention_mask"].to(self.device),
                        labels=inputs['labels'].to(self.device))
        logits = outputs.logits  # (B, L, V)
        labels = inputs['labels'].to(self.device)
        projected_patch_labels = torch.full(
            (inputs['labels'].shape[0], 256),
            fill_value=-100,
            dtype=inputs['labels'].dtype,
            device=self.device,
        )

        multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

        labels = multimodal_labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        

        loss_mask = labels != -100
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps * loss_mask
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        logits_to_keep,
    ):
        pixel_values = pixel_values.to(model.device)
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        ).logits  # (B, L, V)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, -logits_to_keep:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits = logits[:, -logits_to_keep:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    def _save_metrics_to_json(self, metrics, filename):
        """
        将训练指标保存为JSON文件
        
        Args:
            metrics: 要保存的指标字典
            filename: 保存的文件名
        """
        # 确保metrics_dir存在
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 将numpy数组转换为列表以便JSON序列化
        metrics_copy = copy.deepcopy(metrics)
        for key, value in metrics_copy.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (list, np.ndarray)):
                        metrics_copy[key][subkey] = [float(item) if isinstance(item, (np.number, float, int)) else item for item in subvalue]
            elif isinstance(value, (list, np.ndarray)):
                metrics_copy[key] = [float(item) if isinstance(item, (np.number, float, int)) else item for item in value]
        
        # 保存为JSON文件
        filepath = os.path.join(self.metrics_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_copy, f, indent=2)
        
        print(f"指标已保存到 {filepath}")
    
    def train_task(self, task_name, num_steps=10,iteration=1):
        """
        在特定任务上训练模型指定的步数
        
        这是基于步时间(step-wise)的GRPO实现，
        在每个单独的时间步骤内计算优势并更新模型。
        
        Args:
            task_name: 要训练的任务名称
            num_steps: 要执行的训练步数
            
        Returns:
            训练指标字典
        """
        environments = self.task_manager.get_task_environments(task_name)
        metrics = {
            "loss": [],
            "reward": [],
            "kl": [],
            "timestamps": [],  # 添加时间戳记录
            "step_indices": [],  # 添加步骤索引
            "avg_logp": [],      # 添加平均 token 对数概率
            "policy_entropy": [] # 添加策略熵
        }
        
        # 设置当前任务名称
        self.current_task_name = task_name
        
        # 添加进度条显示训练步骤
        step_pbar = tqdm(range(1, num_steps + 1), desc="Training", unit="step")

        
        for step in step_pbar:
            # 记录当前时间戳
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 初始化所有环境
            all_trajectory_data = []
            
            # 为环境添加进度条
            env_pbar = tqdm(enumerate(environments), desc=f"Environments", total=len(environments), leave=False)
            
            # 1. 重置所有环境，准备新的轨迹集合
            for env_idx, env in env_pbar:
                env_pbar.set_description(f"Env {env_idx+1}/{len(environments)}")
                
                # 获取观察
                observation = env.reset()
                
                # 初始化轨迹数据结构
                trajectory_data = {
                    "observations": [observation],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "generated_ids": [],
                    "input_ids": [],
                    "pixel_values": [],
                    "attention_masks": []
                }
                all_trajectory_data.append(trajectory_data)
            
            # 关闭环境进度条
         
            env_pbar.close()
            
            # 2. 执行基于步时间的GRPO：在每个时间步中计算所有环境的优势并更新
            update_group = []
            has_done = False
            a_step_loss = []
            a_step_reward = []
            a_step_kl = []
            a_step_policy_loss = []
            a_step_avg_logp = []
            a_step_entropy = []
            
            for t in tqdm(range(self.config.steps_per_rollout), desc="Step-", leave=False):
                # 阶段1：让所有环境执行一步并收集数据
                step_data_collection = []
                all_terminated = True
                
                if has_done:
                    break
                if t > 1 and True in [trajectory["dones"][-1] for trajectory in all_trajectory_data]:
                    has_done = True
                # 首先让所有环境执行一步并收集数据
                for env_idx, (env, trajectory) in enumerate(zip(environments, all_trajectory_data)):
                    # 检查环境是否已终止

                    # if len(trajectory["dones"]) > 0 and trajectory["dones"][-1]:
                    #     continue
                    
                    all_terminated = False
                    
                    observation = trajectory["observations"][-1]
                    
                    # 准备模型输入
                    inputs,texts = self._prepare_model_inputs(observation, env)
                    
                    # 生成动作
                    action_dim = self.model.get_action_dim(unnorm_key='libero_object')
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=action_dim)
                    
                    action_sequences = self.action_tokenizer.decode_token_ids_to_actions(generated_ids[0,-action_dim:].cpu().numpy())
                    action_norm_stats = self.model.get_action_stats(unnorm_key='libero_object')
                    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                    actions = np.where(
                        mask,
                        0.5 * (action_sequences + 1) * (action_high - action_low) + action_low,
                        action_sequences,
                    )

                    actions = normalize_gripper_action(actions, binarize=True)
                    actions = invert_gripper_action(actions)
                    _data = {'pixel_values':inputs['pixel_values'],'action':generated_ids[0,-action_dim:].cpu().numpy(),'instruction':texts}
                    _data = self._transform(_data)
                    # 在环境中执行动作
                    next_obs, reward, done, info = env.step(actions.tolist())
                    
                    # 记录数据
                    trajectory["observations"].append(next_obs)
                    trajectory["actions"].append(actions)
                    trajectory["rewards"].append(reward)
                    trajectory["dones"].append(done)
                    trajectory["input_ids"].append(_data['input_ids'])
                    trajectory["pixel_values"].append(_data['pixel_values'].to(torch.bfloat16))
                    trajectory["attention_masks"].append(_data['attention_mask'])
                    trajectory["generated_ids"].append(_data['labels'])
                    
                    # 检查是否完成任务并记录成功状态
                    if done:
                        episodes_completed += 1
                        # # 检查是否成功完成任务 (根据Libero环境的info字典)
                        # if 'success' in info and info['success']:
                        #     episodes_succeeded += 1
                    
                    # 收集当前步的数据
                    step_data_collection.append({
                        "env_idx": env_idx,
                        "reward": reward,
                        "inputs": _data,
                        "generated_ids": generated_ids,
                        "attention_mask": inputs["attention_mask"],
                        "reward_stage": env.env.reward_stage,
                        "done": done,
                        "success": info.get('success', False) if done else False
                    })
                update_group.append(step_data_collection)
                # 如果所有环境都已终止，则跳出当前轨迹
                if all_terminated:
                    break
                
                # 阶段2：基于收集的数据执行GRPO更新
            if update_group:
                # 从收集的数据中提取奖励
                
                step_policy_losses = []
                step_kl_losses = []
                n_up = 0
                step_loss = 0
                total_loss = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
                traj_reward = torch.tensor(torch.zeros(4), device=self.device, dtype=torch.bfloat16)
                for step_data_collection in update_group:
                    step_rewards = [data["reward"] for data in step_data_collection]
                    # 计算当前步的平均奖励和标准差
                    step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.bfloat16, device=self.device)
                    traj_reward += step_rewards_tensor
                traj_mean = traj_reward.mean()
                traj_std = traj_reward.std()
                traj_advantages = (traj_reward - traj_mean) / (traj_std + 1e-4)
                for step_data_collection in update_group:
                    n_up += 1
                    step_rewards = [data["reward"] for data in step_data_collection]
                    # 计算当前步的平均奖励和标准差
                    step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.bfloat16, device=self.device)
                    step_mean = step_rewards_tensor.mean()
                    step_std = step_rewards_tensor.std()
                    
                    # 计算当前步的优势值 (群体相对奖励)
                    step_advantages = (step_rewards_tensor - step_mean) / (step_std + 1e-4)
                    
                    step_a_loss = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
                    for i, data in enumerate(step_data_collection):
                        advantage = self.config.a1*step_advantages[i]+ self.config.a2*traj_advantages[i]

                        t_inputs = data["inputs"]
                        
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logp = self._compute_sequence_logp(self.model, t_inputs)
                            
                            
                        with torch.no_grad():
                            ref_logp = self._compute_sequence_logp(self.ref_model, t_inputs)
                        
                        logp_diff = ref_logp - logp
                        kl = torch.exp(logp_diff) - logp_diff - 1

                        policy_loss = torch.exp(logp - logp.detach()) * advantage

                        if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
                            print("警告: 检测到policy_loss中有NaN或Inf值，将其替换为0")
                            raise ValueError("policy_loss中有NaN或Inf值")
                        
                        step_loss = -(policy_loss - self.config.beta * kl).mean()


                        step_a_loss += step_loss


                        metrics["loss"].append(step_loss.item())
                        metrics["reward"].append(step_mean.item())
                        metrics["kl"].append(kl.mean().item())
                        metrics["timestamps"].append(current_time)  # 记录时间戳
                        metrics["step_indices"].append(step)  # 记录步骤索引
                        
                        # 更新步骤进度条信息和wandb记录
                        step_pbar.set_postfix(
                            logggg=f"{policy_loss.mean().item():.4f}",
                            loss=f"{step_loss.item():.4f}", 
                            reward=f"{step_mean.item():.4f}", 
                            kl=f"{kl.mean().item():.4f}"
                        )
                        if self.config.use_wandb:
                            a_step_loss.append(step_loss.item())
                            a_step_reward.append(step_mean.item())
                            a_step_kl.append(kl.mean().item())
                            a_step_policy_loss.append(policy_loss.mean().item())
                            # 计算平均 token 对数概率
                            labels = t_inputs["labels"].to(self.device)
                            projected_patch_labels = torch.full(
                                    (labels.shape[0], 256),
                                    fill_value=-100,
                                    dtype=labels.dtype,
                                    device=self.device,
                                )

                            multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
                            mask = multimodal_labels != -100
                            mask.to(self.device)
                            avg_logp = logp.mean()
                            # 计算策略熵
                            with torch.no_grad():
                                out = self.model(
                                    input_ids=t_inputs["input_ids"].to(self.device),
                                    attention_mask=t_inputs["attention_mask"].to(self.device),
                                    pixel_values=t_inputs["pixel_values"].to(self.device)
                                )
                                lp = F.log_softmax(out.logits, dim=-1)
                                p = lp.exp()
                                per_token_entropy = -(p * lp).sum(dim=-1)
                            policy_entropy = (per_token_entropy*mask).mean()
                            a_step_avg_logp.append(avg_logp.item())
                            a_step_entropy.append(policy_entropy.item())
                            wandb.log({
                                f"{self.current_task_name}/step_reward": step_mean.item(),
                                f"{self.current_task_name}/step_index": i*step*iteration,
                                f"{self.current_task_name}/env_{i}_avg_logp": avg_logp.item(),
                                f"{self.current_task_name}/env_{i}_policy_entropy": policy_entropy.item()
                            })
                    step_a_loss = step_a_loss / self.config.num_environments        
                    self.optimizer.zero_grad()
                    step_a_loss.backward()
                    self.optimizer.step()

            
            if self.config.use_wandb:
                wandb.log({
                    f"{self.current_task_name}/loss": np.mean(a_step_loss),
                    f"{self.current_task_name}/reward": np.mean(a_step_reward),
                    f"{self.current_task_name}/kl": np.mean(a_step_kl),
                    f"{self.current_task_name}/policy_loss": np.mean(a_step_policy_loss),
                    f"{self.current_task_name}/avg_logp": np.mean(a_step_avg_logp),
                    f"{self.current_task_name}/policy_entropy": np.mean(a_step_entropy),
                    "epoch": step*iteration
                })    
            
                # log per-trajectory total rewards for each environment
                episode_returns = [sum(traj["rewards"]) for traj in all_trajectory_data]
                for env_idx, total_return in enumerate(episode_returns):
                    wandb.log({f"{self.current_task_name}/episode_return_env{env_idx}": total_return, "epoch": step*iteration})
        
        # 关闭步骤进度条
        step_pbar.close()
        
        # 保存任务训练指标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_filename = f"{task_name}_metrics_final_{timestamp}.json"
        
        # 添加成功率指标
        metrics["success_rate"] = success_rate
        metrics["episodes_completed"] = episodes_completed
        metrics["episodes_succeeded"] = episodes_succeeded
        
        self._save_metrics_to_json(metrics, metrics_filename)
        
        return metrics
    
    def train(self, num_steps_per_task=100, iterations=1):
        """
        在所有任务上训练模型多次迭代
        
        Args:
            num_steps_per_task: 每次迭代每个任务的训练步数
            iterations: 循环所有任务的迭代次数
        """
        self.model.train()
        
        all_metrics = {task: {
            "loss": [], 
            "reward": [], 
            "kl": [],
            "timestamps": [],  # 添加时间戳记录
            "step_indices": [],  # 添加步骤索引
            "iteration": [],  # 添加迭代索引
            "success_rates": [],  # 添加任务成功率
            "avg_logp": [],  # 添加平均 token 对数概率
            "policy_entropy": []  # 添加策略熵
        } for task in self.config.tasks}
        
        # 记录训练开始时间
        training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_metadata = {
            "start_time": training_start_time,
            "config": {k: str(v) for k, v in vars(self.config).items() if not k.startswith('_')},
            "tasks": self.config.tasks
        }
        
        # 保存训练元数据
        self._save_metrics_to_json(training_metadata, "training_metadata.json")
        print('--------------------------------')
        print('a1',self.config.a1)
        print('a2',self.config.a2)
        print('--------------------------------')
        try:
            # 为迭代添加进度条
            iter_pbar = tqdm(range(1, iterations + 1), desc="Training iterations", unit="iter")
            
            for iteration in iter_pbar:
                iter_pbar.set_description(f"Iteration {iteration}/{iterations}")
                
                # 为任务添加进度条
                task_pbar = tqdm(enumerate(self.config.tasks), total=len(self.config.tasks), 
                                desc="Tasks", unit="task")
                
                for task_idx, task_name in task_pbar:
                    task_pbar.set_description(f"Task {task_idx+1}/{len(self.config.tasks)}: {task_name}")
                    
                    # 在此任务上训练指定的步数
                    task_metrics = self.train_task(
                        task_name=task_name,
                        num_steps=num_steps_per_task,
                        iteration=iteration
                    )
                    
                    # 累积指标
                    for metric_name, values in task_metrics.items():
                        if metric_name in ["step_indices", "success_rate", "episodes_completed", "episodes_succeeded"]:
                            if metric_name == "step_indices":
                                # 调整步骤索引以反映整体训练进度
                                adjusted_indices = [idx + (iteration-1)*num_steps_per_task for idx in values]
                                all_metrics[task_name][metric_name].extend(adjusted_indices)
                            elif metric_name == "success_rate":
                                all_metrics[task_name]["success_rates"].append(values)
                        else:
                            all_metrics[task_name][metric_name].extend(values)
                    
                    # 为每个指标添加当前迭代号
                    iteration_markers = [iteration] * len(task_metrics["loss"])
                    all_metrics[task_name]["iteration"].extend(iteration_markers)
                    
                    # 在每个任务完成后保存该任务的累积指标
                    task_progress_filename = f"{task_name}_progress_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self._save_metrics_to_json(all_metrics[task_name], task_progress_filename)
                    
                    # 更新任务进度条的状态信息
                    avg_loss = np.mean(task_metrics["loss"])
                    avg_reward = np.mean(task_metrics["reward"])
                    task_pbar.set_postfix(
                        loss=f"{avg_loss:.4f}", 
                        reward=f"{avg_reward:.4f}", 
                        success=f"{task_metrics.get('success_rate', 0):.1f}%"
                    )
                
                # 每个迭代结束后保存所有任务的累积指标
                iter_metrics_filename = f"all_tasks_metrics_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self._save_metrics_to_json(all_metrics, iter_metrics_filename)
                
                # 保存LoRA权重 - 在每个迭代结束后保存
                lora_save_dir = os.path.join(self.config.output_dir, f"lora_iter{iteration}")
                os.makedirs(lora_save_dir, exist_ok=True)
                print(f"保存迭代 {iteration} 的LoRA权重到 {lora_save_dir}")
                self.model.save_pretrained(lora_save_dir)
                
                
                # 关闭任务进度条
                task_pbar.close()
                
                # 更新迭代进度条信息
                iter_pbar.set_postfix(checkpoint=iter_metrics_filename)
            
            # 关闭迭代进度条
            iter_pbar.close()
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            raise e
        finally:
            # 记录训练结束时间
            training_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_metadata["end_time"] = training_end_time
            training_metadata["duration"] = str(datetime.strptime(training_end_time, "%Y-%m-%d %H:%M:%S") - 
                                             datetime.strptime(training_start_time, "%Y-%m-%d %H:%M:%S"))
            self._save_metrics_to_json(training_metadata, "training_metadata_final.json")
            
            # 保存最终模型（使用LoRA正确保存）
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 创建临时目录来保存adapter权重
            adapter_dir = os.path.join(self.config.output_dir, "adapter_tmp")
            os.makedirs(adapter_dir, exist_ok=True)
            self.task_manager.close_all()
            # 先保存adapter权重
            print("保存LoRA adapter权重...")
            self.model.save_pretrained(adapter_dir)
            # 保存tokenizer
            
            # 将模型从GPU卸载到CPU，释放GPU内存
            print("将模型从GPU卸载到CPU...")
            self.ref_model.to("cpu")
            del self.ref_model
            # 加载基础模型
            print("加载基础模型...")
            base_model = AutoModelForVision2Seq.from_pretrained(
                "/home/vipuser/hdp/openvla-7b-finetuned-libero-object", 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization_config=None
            )
            
            # 将adapter权重加载到基础模型
            print("合并LoRA权重到基础模型...")
            merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
            
            # 合并权重（这将把LoRA权重合并到基础模型中）
            merged_model = merged_model.merge_and_unload()
            
            # 保存合并后的完整模型
            print(f"保存合并后的完整模型到 {self.config.output_dir}...")
            merged_model.save_pretrained(self.config.output_dir)
            
            # 保存完整训练指标
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_metrics_filename = f"all_metrics_final_{timestamp}.json"
            self._save_metrics_to_json(all_metrics, all_metrics_filename)
            
            # 关闭所有环境
            
        
        return all_metrics


if __name__ == "__main__":
    # 示例用法
    model_path = "/home/vipuser/hdp/openvla-7b-finetuned-libero-object"  # 替换为你的OpenVLA模型路径
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained('/home/vipuser/hdp/openvla-7b-finetuned-libero-object', trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=None
    )

    # 方法1：从Libero的特定任务ID创建训练器
    # 这将自动将任务ID转换为任务名称
    trainer = StepWiseGRPOTrainer.from_task_ids(
        model=model,
        processor=processor,
        task_suite_name="libero_object",
        task_ids=[9],  # task_ids=[0,1,2,3,4,5,6,7,8,9]
        output_dir="stepwise_grpo_output_tasks_all",
        learning_rate=1e-5,
        a1=0.3,
        a2=0.7,
        logging_steps=5,
        batch_size=4,
        num_environments=4,
        steps_per_rollout=200,
        beta=0.01,
        use_enhanced_rewards=True,   # 启用增强奖励
        reward_scale=0.5,            # 密集奖励的比例因子
        log_rewards=True,            # 记录详细的奖励信息
        use_wandb=True,
        wandb_project="libero_grpo",
        wandb_run_name="task_9_hyper_param_test"
    )
    
    # 开始多任务训练 - 每个任务50步，所有任务2次迭代
    print(f"Training on tasks: {trainer.config.tasks}")
    metrics = trainer.train(num_steps_per_task=1, iterations=1) 