# config/training_args.py
import os
from dataclasses import dataclass

@dataclass
class TrainingArguments:
    # Model & Tokenizer
    base_tokenizer: str = "Qwen/Qwen1.5-7B-Chat" # A modern, efficient tokenizer
    
    # Dataset Names
    pretrain_dataset_cosmopedia: str = "HuggingFaceTB/cosmopedia"
    pretrain_dataset_cosmopedia_subset: str = "web_samples_v1"
    pretrain_dataset_numinamath: str = "AI-MO/NuminaMath-CoT"
    finetune_dataset_ultrachat: str = "stingning/ultrachat"
    
    # System Prompt
    system_prompt: str = (
        "You are ProbSolSpace 1.0 Transformer 50M, an experimental language model "
        "trained by Jasper Jiang under the Dawntasy initiative. You are a highly capable "
        "and intelligent AI assistant designed to excel in problem solving, reasoning, "
        "creativity and instruction-following. The user has provided a prompt. Please "
        "reply in a respectful, knowledgeable and accurate manner."
    )

    # Training Hyperparameters
    learning_rate: float = 3e-4
    max_seq_length: int = 2048
    per_device_batch_size: int = 4 # Will result in global batch size of 8 * 4 = 32
    gradient_accumulation_steps: int = 4 # Effective batch size = 32 * 2 = 64
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    grad_clip: float = 1.0

    # Stage-specific Steps
    pretrain_steps: int = 20000 # Number of steps for pre-training phase
    finetune_steps: int = 5000   # Number of steps for fine-tuning phase

    # Logging and Saving
    logging_steps: int = 20
    output_dir: str = "./results"
    final_model_dir: str = os.path.join(output_dir, "final_model")
    metrics_report_path: str = os.path.join(output_dir, "training_report.png")
