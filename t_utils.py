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
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from tqdm.auto import tqdm  # 导入tqdm进度条库
# Import enhanced reward wrapper modules
from enhanced_reward_wrapper import EnhancedRewardWrapper, create_env_with_enhanced_rewards
import custom_reward_functions
from peft import LoraConfig, get_peft_model
@dataclass
class OpenVLAGRPOConfig:
    output_dir: str
    learning_rate: float = 1e-5
    logging_steps: int = 10
    batch_size: int = 16
    max_completion_length: int = 32   # Maximum tokens to generate
    num_environments: int = 4         # Number of parallel environments per task
    steps_per_rollout: int = 200      # Number of steps to execute in each environment
    beta: float = 0.1                 # KL penalty coefficient
    tasks: List[str] = None           # List of task names to train on
    task_suite_name: str = "libero_object"  # Name of the task suite
    camera_height: int = 256          # Camera height for Libero environment
    camera_width: int = 256           # Camera width for Libero environment
    image_size: Tuple[int, int] = (224, 224)  # Size of images for OpenVLA model
    use_enhanced_rewards: bool = True  # Whether to use enhanced rewards
    reward_scale: float = 0.1         # Scale factor for enhanced rewards
    log_rewards: bool = False         # Whether to log reward information
    use_wandb: bool = False           # Whether to use wandb
    wandb_project: str = "libero_grpo" # Wandb project name
    wandb_run_name: str = "libero_grpo"        # Wandb entity name
    a1: float = 1.0
    a2: float = 0.1


class RobotEnvironment:
    """
    Robot environment interface class that wraps Libero's OffScreenRenderEnv.
    
    This class:
    - Creates a Libero environment for a specific task
    - Directly passes through the observations, rewards, and states from Libero
    """
    def __init__(self, env_id, task_name, task_suite_name="libero_object", camera_height=128, camera_width=128, image_size=(224, 224), use_enhanced_rewards=True, reward_scale=0.1, log_rewards=False):
        self.env_id = env_id
        self.task_name = task_name
        self.step_count = 0
        self.image_size = image_size
        self.use_enhanced_rewards = use_enhanced_rewards
        self.reward_scale = reward_scale
        self.log_rewards = log_rewards
        
        # Set up Libero environment
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[task_suite_name]()
        
        # Find the task id based on task_name
        self.task_id = None
        for i in range(len(self.task_suite.tasks)):
            if self.task_suite.tasks[i].name == task_name:
                self.task_id = i
                break
        
        if self.task_id is None:
            # If task_name doesn't match exactly, use task_id based on env_id
            self.task_id = env_id % len(self.task_suite.tasks)
            print(f"Warning: Task {task_name} not found. Using task_id {self.task_id} instead.")
        
        # Get task description directly from tasks list
        self.task_description = self.task_suite.tasks[self.task_id].language
        self.task = self.task_suite.tasks[self.task_id]
        self.task_bddl_file = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        
        print(f"[info] Initializing environment {env_id} with task {self.task_id}, name: {self.task.name}")
        print(f"[info] Language instruction: {self.task_description}")
        
        # Create the environment
        env_args = {
            "bddl_file_name": self.task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        
        # Create base environment or enhanced environment
        if self.use_enhanced_rewards:
            # Extract task_id from BDDL file name for better reward function matching
            bddl_task_id = os.path.basename(self.task_bddl_file)
            if bddl_task_id.endswith('.bddl'):
                bddl_task_id = bddl_task_id[:-5]  # Remove .bddl extension
                
            # Create environment with enhanced rewards
            self.env = create_env_with_enhanced_rewards(
                bddl_file=self.task_bddl_file,
                task_name=self.task.name,
                task_id=bddl_task_id,
                task_description=self.task_description,
                reward_scale=self.reward_scale,
                env_args=env_args,
                log_rewards=self.log_rewards
            )
            print(f"[info] Using enhanced rewards for task {self.task.name}")
        else:
            # Create base environment without enhanced rewards
            self.env = OffScreenRenderEnv(**env_args)
            
        self.env.seed(env_id)  # Use env_id as seed for reproducibility
        
        # Get init states for the task
        self.init_states = self.task_suite.get_task_init_states(self.task_id)
        if len(self.init_states) > 0:
            self.init_state_id = env_id % len(self.init_states)
        else:
            self.init_state_id = None
    
    def reset(self):
        """Reset environment and directly return the original observation"""
        self.step_count = 0
        return self.env.reset()
    
    def step(self, action):
        """
        Execute action and directly return the original observation, reward, done, info
        
        Args:
            action: Action to execute (text or parsed commands)
            
        Returns:
            observation: Original observation from Libero
            reward: Reward from Libero (enhanced if enabled)
            done: Whether episode is done
            info: Additional information
        """
        self.step_count += 1
        
        # Convert text action to action vector for Libero
        action_vector = action
        
        # Execute action in the Libero environment and return results directly
        obs, reward, done, info = self.env.step(action_vector)
        
        # Add basic tracking info without modifying the observation
        info["step_count"] = self.step_count
        info["task_id"] = self.task_id
        
        return obs, reward, done, info
    
    # def _parse_action_to_vector(self, action_text):
    #     """
    #     Parse text action into a 7-dimensional vector for Libero.
        
    #     In a real implementation, this should be a more sophisticated parsing of the 
    #     action text generated by OpenVLA to properly control the Libero robot.
        
    #     For this example implementation, we'll use a very simple approach that extracts
    #     numbers from the text or returns a default action.
    #     """
    #     try:
    #         # Simple example of parsing - extract numbers from the action text
    #         import re
    #         numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", action_text)
            
    #         if len(numbers) >= 7:
    #             # If we have enough numbers, use them as the action
    #             return [float(num) for num in numbers[:7]]
    #         elif len(numbers) > 0:
    #             # If we have some numbers but not enough, repeat the last one
    #             values = [float(num) for num in numbers]
    #             while len(values) < 7:
    #                 values.append(values[-1])
    #             return values
    #         else:
    #             # Default action - small random movement
    #             return [0.05 * np.random.randn() for _ in range(7)]
    #     except Exception as e:
    #         print(f"Error parsing action: {e}")
    #         # Fallback to a default action
    #         return [0.0] * 7
    
    def close(self):
        self.env.close()


class TaskManager:
    """
    Manages multiple tasks and environments for multi-task reinforcement learning.
    Each task has multiple parallel environments.
    """
    def __init__(self, tasks: List[str], num_envs_per_task: int, task_suite_name="libero_object", 
                 camera_height=128, camera_width=128, image_size=(224, 224), 
                 use_enhanced_rewards=True, reward_scale=0.1, log_rewards=False):
        self.tasks = tasks
        self.num_envs_per_task = num_envs_per_task
        self.image_size = image_size
        self.task_suite_name = task_suite_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.use_enhanced_rewards = use_enhanced_rewards
        self.reward_scale = reward_scale
        self.log_rewards = log_rewards
        
        # Create environments for each task
        self.task_environments = {}
        
        for task_name in tasks:
            self.task_environments[task_name] = [
                RobotEnvironment(
                    env_id=env_id, 
                    task_name=task_name, 
                    task_suite_name=task_suite_name,
                    camera_height=camera_height,
                    camera_width=camera_width,
                    image_size=image_size,
                    use_enhanced_rewards=use_enhanced_rewards,
                    reward_scale=reward_scale,
                    log_rewards=log_rewards
                )
                for env_id in range(num_envs_per_task)
            ]
    
    @classmethod
    def from_task_suite(cls, task_suite_name="libero_10", num_envs_per_task=4, 
                        camera_height=128, camera_width=128, image_size=(224, 224),
                        task_ids=None, use_enhanced_rewards=True, reward_scale=0.1, log_rewards=False):
        """
        Create TaskManager from a Libero task suite, optionally filtering to specific task IDs.
        
        Args:
            task_suite_name: Name of the Libero task suite
            num_envs_per_task: Number of environments to create per task
            camera_height: Camera height for Libero environments
            camera_width: Camera width for Libero environments
            image_size: Size of images for OpenVLA model
            task_ids: Optional list of task IDs to include (if None, include all tasks)
            use_enhanced_rewards: Whether to use enhanced rewards
            reward_scale: Scale factor for enhanced rewards
            log_rewards: Whether to log reward information
            
        Returns:
            Initialized TaskManager
        """
        # Get all tasks from the task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        # Get task names from the suite
        if task_ids is None:
            # Use all tasks
            task_ids = list(range(len(task_suite.tasks)))
        
        task_names = [task_suite.tasks[task_id].name for task_id in task_ids]
        
        # Create manager
        return cls(
            tasks=task_names,
            num_envs_per_task=num_envs_per_task,
            task_suite_name=task_suite_name,
            camera_height=camera_height,
            camera_width=camera_width,
            image_size=image_size,
            use_enhanced_rewards=use_enhanced_rewards,
            reward_scale=reward_scale,
            log_rewards=log_rewards
        )
    
    def get_task_environments(self, task_name):
        """Get all environments for a specific task"""
        if task_name not in self.task_environments:
            raise ValueError(f"Task {task_name} not found in task manager")
        return self.task_environments[task_name]
    
    def get_all_tasks(self):
        """Get list of all task names"""
        return list(self.task_environments.keys())
    
    def close_all(self):
        """Close all environments"""
        for task_name in self.task_environments:
            for env in self.task_environments[task_name]:
                env.close()


# class OpenVLAGRPOTrainer:
#     """
#     GRPO Trainer for OpenVLA models with multi-task and multi-environment parallel training.

#     This trainer implements GRPO for vision-language models:
#     1. Creates multiple tasks, each with multiple parallel environments
#     2. For each training step:
#        - Selects a task and resets its environments
#        - Generates action sequences for the task based on the same instruction + environment observations
#        - Executes each action sequence across environments for steps_per_rollout steps
#        - Collects cumulative rewards from each environment
#        - Computes advantages using the group (environments) mean/std within the task
#        - Calculates GRPO loss with KL penalty from reference model
#     """
#     def __init__(self, model, processor, config: OpenVLAGRPOConfig):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device)
#         lora_config = LoraConfig(
#             r=8,
#             lora_alpha=32,
#             target_modules=["q_proj", "v_proj"],  # 根据模型具体情况修改
#             lora_dropout=0.1,
#             bias="none"
#         )
#         self.model = get_peft_model(self.model, lora_config)
#         params = [param for param in self.model.parameters() if param.requires_grad]
#         # Create a frozen reference model
#         self.ref_model=AutoModelForVision2Seq.from_pretrained(
#                 "/home/vipuser/hdp/openvla-7b-finetuned-libero-object",
#                 torch_dtype=torch.float16,
#                 trust_remote_code=True,
#                 quantization_config=None
#             ).to(self.device)

#         self.ref_model.eval()
#         for param in self.ref_model.parameters():
#             param.requires_grad = False
        
#         self.processor = processor
#         self.config = config
#         self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
#         self.base_tokenizer = self.processor.tokenizer
#         self.prompt_builder = PurePromptBuilder("openvla")
#         # Initialize current_task_name attribute
#         self.current_task_name = None
        
#         # Set up tasks and task manager
#         if self.config.tasks is None:
#             # Get all tasks from the task suite
#             benchmark_dict = benchmark.get_benchmark_dict()
#             task_suite = benchmark_dict[config.task_suite_name]()
#             self.config.tasks = [task_suite.get_task(i).name for i in range(len(task_suite.tasks))]
#             print(f"[info] Using all tasks from {config.task_suite_name}: {self.config.tasks}")
        
#         # Create task manager with environments
#         self.task_manager = TaskManager(
#             tasks=self.config.tasks,
#             num_envs_per_task=self.config.num_environments,
#             task_suite_name=self.config.task_suite_name,
#             camera_height=self.config.camera_height,
#             camera_width=self.config.camera_width,
#             image_size=self.config.image_size,
#             use_enhanced_rewards=self.config.use_enhanced_rewards,
#             reward_scale=self.config.reward_scale,
#             log_rewards=self.config.log_rewards
#         )
        
#         # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
#         self.optimizer = optim.AdamW(params, lr=self.config.learning_rate)
#         self.generation_config = GenerationConfig(
#             max_new_tokens=self.config.max_completion_length,
#             do_sample=True,
#             temperature=1.0,
#             pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
#         )
    
#     @classmethod
#     def from_task_ids(cls, model, processor, task_suite_name="libero_object", task_ids=None, 
#                      output_dir="openvla_grpo_output", **kwargs):
#         """
#         Create trainer directly from task IDs.
        
#         Args:
#             model: The OpenVLA model to fine-tune
#             processor: The OpenVLA processor
#             task_suite_name: Name of the Libero task suite
#             task_ids: List of task IDs to include (if None, include all tasks)
#             output_dir: Directory to save outputs
#             **kwargs: Additional arguments for OpenVLAGRPOConfig
            
#         Returns:
#             Initialized OpenVLAGRPOTrainer
#         """
#         # Get task names from IDs
#         benchmark_dict = benchmark.get_benchmark_dict()
#         task_suite = benchmark_dict[task_suite_name]()
        
#         if task_ids is None:
#             task_ids = list(range(len(task_suite.tasks)))
            
#         task_names = [task_suite.tasks[task_id].name for task_id in task_ids]
        
#         # Create config
#         config = OpenVLAGRPOConfig(
#             output_dir=output_dir,
#             tasks=task_names,
#             task_suite_name=task_suite_name,
#             **kwargs
#         )
        
#         return cls(model=model, processor=processor, config=config)
    
#     def _prepare_model_inputs(self, observations,env):
#         """
#         Process Libero observations to create inputs for the OpenVLA model.
        
#         Args:
#             observations: List of original Libero observations (containing 'agentview_image' key)
            
#         Returns:
#             Dictionary of tensors ready for model input
#         """
#         # Extract images from Libero observations
#         images = observations["agentview_image"]
#         prompt_builder = PurePromptBuilder("openvla")
#         # Create text inputs using task descriptions
#         texts = env.task_description
#         # print('==========task description==========')
#         # print(texts)
#         # print('====================================')

#         prompt_builder.add_turn(role="human", message=f"What action should the robot take to {texts}?")

#         # Tokenize (w/ `base_tokenizer`)
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         attention_mask = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).attention_mask
#         attention_mask = torch.tensor([attention_mask])
#         input_ids = torch.tensor([input_ids])
#         # Process inputs with the processor
#         if isinstance(images, np.ndarray):
#            images = Image.fromarray(images.astype(np.uint8))
#         pixel_values = self.processor.image_processor.apply_transform(images)
#         pixel_values = torch.tensor(pixel_values).unsqueeze(0).to(torch.float16)

#         inputs = {
#             "pixel_values": pixel_values,
#             "input_ids": input_ids,
#             "attention_mask": attention_mask
#         }

#         for k, v in inputs.items():
#             inputs[k] = v.to(self.device)
                
#         return inputs
    
#     def _compute_sequence_logp(self, model, inputs, generated_ids,num_device=0):
#         """
#         Compute the sequence-level log-probabilities for generated tokens.
        
#         Args:
#             model: The model to compute log probs
#             inputs: Model inputs including pixel_values, input_ids, etc.
#             generated_ids: (B, L) tensor of generated token ids
            
#         Returns:
#             (B,) tensor of log probabilities
#         """
#         # Prepare inputs for the model - we need to calculate the log probability of the generated sequence
#         # For this, we replace the input_ids with generated_ids and keep the images
#         model_inputs = {
#             "pixel_values": inputs["pixel_values"],
#             "input_ids": generated_ids,
#             "attention_mask": (generated_ids != self.processor.tokenizer.pad_token_id).long()
#         }
#         # if num_device == 1:
#         #     for k, v in model_inputs.items():
#         #         model_inputs[k] = v.to(self.device2)
#         outputs = model(**model_inputs)

#         logits = outputs.logits  # (B, L, V)
#         # logits = logits.to(self.device)
#         # Compute log probs by shifting predictions and targets
#         # Model predicts next token, so we shift to align
#         shifted_logits = logits[:, :-1, :]  # (B, L-1, V)
#         shifted_ids = generated_ids[:, 1:]  # (B, L-1)
#         log_probs = F.log_softmax(shifted_logits, dim=-1)  # (B, L-1, V)
        
#         # Gather log probs for the actual generated tokens
#         token_log_probs = torch.gather(
#             log_probs, 
#             dim=2, 
#             index=shifted_ids.unsqueeze(-1)
#         ).squeeze(-1)  # (B, L-1)
        
#         # Mask to exclude padding tokens
#         padding_mask = (shifted_ids != self.processor.tokenizer.pad_token_id).float()
#         masked_log_probs = token_log_probs * padding_mask
        
#         # Sum log probs along sequence dimension to get sequence log prob
#         sequence_logps = masked_log_probs.sum(dim=1)  # (B,)
        
#         return sequence_logps
    
#     def execute_rollout(self, env, action_sequence, steps_to_execute):
#         """
#         Execute a rollout in the environment for a given number of steps.
        
#         Args:
#             env: Environment to execute in
#             action_sequence: Action sequence to execute (text)
#             steps_to_execute: Number of steps to execute
            
#         Returns:
#             total_reward: Total reward accumulated during rollout
#             trajectory: Complete trajectory information
#         """
#         observation = env.reset()
#         total_reward = 0
#         trajectory = {
#             "observations": [observation],
#             "actions": [],
#             "rewards": [],
#             "dones": [],
#             "action_sequence": action_sequence,
#         }
        
#         # Parse action_sequence into individual actions 
#         # In practice, you would implement proper parsing for your specific environment
#         actions = action_sequence.split(".")  # Simple example: split by period
#         actions = [a.strip() for a in actions if a.strip()]
        
#         # Extend actions list if shorter than steps_to_execute by repeating
#         if len(actions) < steps_to_execute:
#             actions = (actions * ((steps_to_execute // len(actions)) + 1))[:steps_to_execute]
        
#         # Execute steps
#         for i in range(steps_to_execute):
#             action = actions[i] if i < len(actions) else "idle"  # Default to idle if out of actions
            
#             # Execute action in environment
#             next_obs, reward, done, info = env.step(action)
            
#             # Record step information
#             trajectory["actions"].append(action)
#             trajectory["rewards"].append(reward)
#             trajectory["dones"].append(done)
#             trajectory["observations"].append(next_obs)
            
#             # Accumulate reward
#             total_reward += reward
            
#             if done:
#                 break
        
#         return total_reward, trajectory
    
#     def train_task(self, task_name, num_steps=10):
#         """
#         Train the model on a specific task for a number of steps.
        
#         Args:
#             task_name: Name of the task to train on
#             num_steps: Number of training steps to perform
            
#         Returns:
#             Dictionary of training metrics
#         """
#         environments = self.task_manager.get_task_environments(task_name)
#         metrics = {
#             "loss": [],
#             "reward": [],
#             "kl": []
#         }
        
#         # Set current task name for use in _prepare_model_inputs
#         self.current_task_name = task_name
        
#         # 添加进度条来显示训练步骤
#         step_pbar = tqdm(range(1, num_steps + 1), desc=f"Training on {task_name}", unit="step")
        
#         for step in step_pbar:
#             all_trajectory_data = []
#             all_rewards = []
            
#             # 为环境添加进度条
#             env_pbar = tqdm(enumerate(environments), desc=f"Environments", total=len(environments), leave=False)
            
#             for env_idx, env in env_pbar:
#                 # 更新环境进度条描述
#                 env_pbar.set_description(f"Env {env_idx+1}/{len(environments)}")
                
#                 # 1. Get observations from environments for this task
#                 observation = env.reset()
#                 trajectory_data = {
#                     "observations": [observation],
#                     "actions": [],
#                     "rewards": [],
#                     "dones": [],
#                     "generated_ids": [],
#                     "input_ids": [],
#                     "pixel_values": [],
#                     "attention_masks": []
#                 }
#                 total_reward = 0
                
#                 # 为每个环境的步骤添加进度条
#                 rollout_pbar = tqdm(range(self.config.steps_per_rollout), 
#                                    desc=f"Rollout", leave=False, unit="step")
                
#                 for t in rollout_pbar:
#                     # 2. Prepare inputs for the model with the processor
#                     inputs = self._prepare_model_inputs(observation,env)
#                     with torch.autocast("cuda", dtype=torch.float16):
#                         generated_ids = self.model.generate(
#                             input_ids=inputs["input_ids"],
#                             pixel_values=inputs["pixel_values"],
#                             attention_mask=inputs["attention_mask"],
#                             max_new_tokens=7)
#                     # print('==========predicted generated_ids==========')
#                     # print(generated_ids)
#                     # print('===========================================')
#                     # 4. Decode action sequences
#                     action_sequences = self.action_tokenizer.decode_token_ids_to_actions(generated_ids[0,-7:].cpu())
#                     mask = np.array([1,1,1,1,1,1,0]).astype(bool)
#                     action_high, action_low = np.array([0.9375,0.9107142686843872,0.9375,0.20357142388820648,0.26357144117355347,0.375,1.0]), np.array([-0.8785714507102966,-0.7553571462631226,-0.9375,-0.1510714292526245,-0.1639285683631897,-0.13777500048279764,0.0])
#                     actions = np.where(
#                         mask,
#                         0.5 * (action_sequences + 1) * (action_high - action_low) + action_low,
#                         action_sequences,
#                     )
#                     # print('==========predicted action==========')
#                     # print(actions)
#                     # print('====================================')
#                     next_obs, reward, done, info = env.step(actions)
#                     trajectory_data["observations"].append(next_obs)
#                     trajectory_data["actions"].append(actions)
#                     trajectory_data["rewards"].append(reward)
#                     trajectory_data["dones"].append(done)
#                     trajectory_data["input_ids"].append(inputs["input_ids"])
#                     trajectory_data["pixel_values"].append(inputs["pixel_values"].to(torch.float16))
#                     trajectory_data["attention_masks"].append(inputs["attention_mask"])
#                     trajectory_data["generated_ids"].append(generated_ids)
                    
#                     total_reward += reward
#                     observation = next_obs
                    
#                     # 更新rollout进度条信息
#                     rollout_pbar.set_postfix(reward=reward, total_reward=total_reward)
                    
#                     if done:
#                         break
                
#                 # 关闭rollout进度条
#                 rollout_pbar.close()
#                 all_trajectory_data.append(trajectory_data)
#                 all_rewards.append(total_reward)
                
#                 # 更新环境进度条信息
#                 env_pbar.set_postfix(total_reward=total_reward)
            
#             # 关闭环境进度条
#             env_pbar.close()
            
#             # 6. Compute advantages for this task group
#             rewards = torch.tensor(all_rewards, dtype=torch.float16, device=self.device)
#             group_mean = rewards.mean()
#             group_std = rewards.std() if rewards.std() > 0 else torch.tensor(1.0, device=self.device)
#             advantages = (rewards - group_mean) / (group_std + 1e-4)
            
#             env_policy_losses = []
#             env_kl_losses = []
            
#             for env_idx, (trajectory, advantage) in enumerate(zip(all_trajectory_data, advantages)):
#                 env_policy_loss = 0
#                 env_kl_loss = 0
#                 for t in range(len(trajectory["actions"])):
#                     t_inputs = {
#                         "input_ids": trajectory["input_ids"][t],
#                         "pixel_values": trajectory["pixel_values"][t],
#                         "attention_mask": trajectory["attention_masks"][t] if trajectory["attention_masks"][t] is not None else None
#                     }
#                     t_generated_ids = trajectory["generated_ids"][t]

#                     with torch.autocast("cuda", dtype=torch.float16):
#                         logp = self._compute_sequence_logp(self.model, t_inputs, t_generated_ids,num_device=0)
                    
#                     with torch.no_grad():
#                         # t_inputs2 = t_inputs.to(self.device2)
#                         # t_generated_ids2 = t_generated_ids.to(self.device2)
#                         ref_logp = self._compute_sequence_logp(self.ref_model, t_inputs, t_generated_ids,num_device=1)
#                         # ref_logp = ref_logp.to(self.device)
#                     # 修复变量名，使用正确的KL计算
#                     kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
#                     env_policy_loss += -(advantage * logp)
#                     env_kl_loss += kl
                
#                 # 计算每个环境的平均损失
#                 steps = len(trajectory["actions"])
#                 if steps > 0:
#                     env_policy_losses.append(env_policy_loss / steps)
#                     env_kl_losses.append(env_kl_loss / steps)
            
#             # 计算所有环境的平均损失
#             policy_loss = torch.stack(env_policy_losses).mean() if env_policy_losses else torch.tensor(0.0)
#             print(f"policy_loss: {policy_loss.item()}")
#             kl_loss = torch.stack(env_kl_losses).mean() if env_kl_losses else torch.tensor(0.0)
#             print(f"kl_loss: {kl_loss.item()}")
            
#             # 计算总损失
#             loss = policy_loss + self.config.beta * kl_loss
            
#             # 更新模型参数
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             # Record metrics
#             metrics["loss"].append(loss.item())
#             metrics["reward"].append(group_mean.item())
#             metrics["kl"].append(kl_loss.item())  # 修正为kl_loss.item()
            
#             # 更新步骤进度条信息
#             step_pbar.set_postfix(loss=f"{loss.item():.4f}", reward=f"{group_mean.item():.4f}", kl=f"{kl_loss.item():.4f}")
        
#         # 关闭步骤进度条
#         step_pbar.close()
#         return metrics
    
#     def train(self, num_steps_per_task=100, iterations=1):
#         """
#         Train the model on all tasks for multiple iterations.
        
#         Args:
#             num_steps_per_task: Number of training steps per task per iteration
#             iterations: Number of iterations to cycle through all tasks
#         """
#         self.model.train()
        
#         all_metrics = {task: {
#             "loss": [], "reward": [], "kl": []
#         } for task in self.config.tasks}
        
#         try:
#             # 为迭代添加进度条
#             iter_pbar = tqdm(range(1, iterations + 1), desc="Training iterations", unit="iter")
            
#             for iteration in iter_pbar:
#                 iter_pbar.set_description(f"Iteration {iteration}/{iterations}")
                
#                 # 为任务添加进度条
#                 task_pbar = tqdm(enumerate(self.config.tasks), total=len(self.config.tasks), 
#                                 desc="Tasks", unit="task")
                
#                 for task_idx, task_name in task_pbar:
#                     task_pbar.set_description(f"Task {task_idx+1}/{len(self.config.tasks)}: {task_name}")
                    
#                     # Train on this task for specified number of steps
#                     task_metrics = self.train_task(
#                         task_name=task_name,
#                         num_steps=num_steps_per_task
#                     )
                    
#                     # Accumulate metrics
#                     for metric_name, values in task_metrics.items():
#                         all_metrics[task_name][metric_name].extend(values)
                    
#                     # Save checkpoint after each task
#                     checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-iter{iteration}-task{task_idx}")
#                     os.makedirs(checkpoint_dir, exist_ok=True)
#                     self.model.save_pretrained(checkpoint_dir)
                    
#                     # 更新任务进度条的状态信息
#                     avg_loss = np.mean(task_metrics["loss"])
#                     avg_reward = np.mean(task_metrics["reward"])
#                     task_pbar.set_postfix(loss=f"{avg_loss:.4f}", reward=f"{avg_reward:.4f}")
                
#                 # 关闭任务进度条
#                 task_pbar.close()
                
#                 # Save checkpoint after full iteration
#                 checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-iter{iteration}")
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 self.model.save_pretrained(checkpoint_dir)
                
#                 # 更新迭代进度条信息
#                 iter_pbar.set_postfix(checkpoint=checkpoint_dir)
            
#             # 关闭迭代进度条
#             iter_pbar.close()
        
#         finally:
#             # Save final model
#             os.makedirs(self.config.output_dir, exist_ok=True)
#             self.model.save_pretrained(self.config.output_dir)
            
#             # Close all environments
#             self.task_manager.close_all()
        
#         return all_metrics


# if __name__ == "__main__":
#     # Example usage
#     model_path = "/home/vipuser/hdp/openvla-7b-finetuned-libero-object"  # Replace with your OpenVLA model path
#     AutoConfig.register("openvla", OpenVLAConfig)
#     AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
#     AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
#     AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
#     # Load model and processor
#     processor = AutoProcessor.from_pretrained('/home/vipuser/hdp/openvla-7b-finetuned-libero-object', trust_remote_code=True)
#     model = AutoModelForVision2Seq.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#         quantization_config=None
#     )
    
#     # Method 1: Create trainer with specific task IDs from Libero
#     # This will automatically convert task IDs to task names
#     trainer1 = OpenVLAGRPOTrainer.from_task_ids(
#         model=model,
#         processor=processor,
#         task_suite_name="libero_object",
#         task_ids=[0, 1, 2],  # Select first three tasks from libero_object
#         output_dir="openvla_grpo_output_tasks_0_1_2",
#         learning_rate=1e-5,
#         logging_steps=5,
#         num_environments=2,
#         steps_per_rollout=5,
#         beta=0.1,
#         use_enhanced_rewards=True,   # Enable enhanced rewards
#         reward_scale=0.1,            # Scale factor for dense rewards
#         log_rewards=True             # Log detailed reward information
#     )
    
#     # # Method 2: Create task manager first, then create trainer with config
#     # task_manager = TaskManager.from_task_suite(
#     #     task_suite_name="libero_object",
#     #     num_envs_per_task=4,
#     #     task_ids=[0, 3, 5],  # Select specific tasks by ID
#     #     use_enhanced_rewards=True,
#     #     reward_scale=0.1,
#     #     log_rewards=True
#     # )
    
#     # # Create config manually
#     # config = OpenVLAGRPOConfig(
#     #     output_dir="openvla_grpo_output",
#     #     learning_rate=1e-5,
#     #     logging_steps=5,
#     #     max_completion_length=32,
#     #     num_environments=4,
#     #     steps_per_rollout=200,
#     #     beta=0.1,
#     #     tasks=task_manager.get_all_tasks(),
#     #     task_suite_name="libero_object",
#     #     camera_height=128,
#     #     camera_width=128,
#     #     image_size=(224, 224),
#     #     use_enhanced_rewards=True,
#     #     reward_scale=0.1,
#     #     log_rewards=True
#     # )
    
#     # # Create trainer with config
#     # trainer2 = OpenVLAGRPOTrainer(
#     #     model=model,
#     #     processor=processor,
#     #     config=config
#     # )
    
#     # Choose which trainer to use
#     trainer = trainer1  # or trainer2
    
#     # Start multi-task training - 50 steps per task, 2 iterations through all tasks
#     print(f"Training on tasks: {trainer.config.tasks}")
#     metrics = trainer.train(num_steps_per_task=50, iterations=2) 