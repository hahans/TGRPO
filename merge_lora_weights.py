import os
import torch
import argparse
from transformers import AutoConfig, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor
from peft import PeftModel
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def merge_lora_weights(
    base_model_path, 
    adapter_path, 
    output_path, 
    save_half=False, 
    device="cuda"
):
    """
    Merge LoRA weights from adapter_path with the base model and save the merged model
    
    Args:
        base_model_path (str): Path to the base model
        adapter_path (str): Path to the LoRA adapter weights
        output_path (str): Path to save the merged model
        save_half (bool): Whether to save the model in half precision
        device (str): Device to use for loading the model
    """
    print(f"Registering OpenVLA model classes...")
    # Register model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading base model from {base_model_path}...")
    
    # Set precision type based on the save_half flag
    if save_half:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
        
    # Load the base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    # Load adapter weights
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=dtype
        )
        print("Successfully loaded LoRA adapter")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        raise
    
    print("Merging LoRA weights into base model...")
    # Merge weights
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    print("Model successfully merged and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="/home/vipuser/hdp/openvla-7b-finetuned-libero-object",
        help="Path to the base model"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="/home/vipuser/hdp/libero_grpo/stepwise_grpo_output_tasks_all/adapter_tmp",
        help="Path to the LoRA adapter weights"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="/home/vipuser/hdp/libero_grpo/stepwise_grpo_output_tasks_all/merged_model",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--save_half", 
        action="store_true",
        help="Save the model in half precision (fp16 instead of bf16)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Add additional checks to avoid OOM errors
    if torch.cuda.is_available() and args.device == "cuda":
        # Free up memory if possible
        torch.cuda.empty_cache()
        
        # Check available GPU memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_gb = free_memory / (1024**3)
        
        print(f"Available GPU memory: {free_memory_gb:.2f} GB")
        
        if free_memory_gb < 10:  # If less than 10GB memory available
            print("WARNING: Low GPU memory available. Consider using CPU or freeing up GPU memory.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting.")
                exit(0)
    
    merge_lora_weights(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        save_half=args.save_half,
        device=args.device
    ) 