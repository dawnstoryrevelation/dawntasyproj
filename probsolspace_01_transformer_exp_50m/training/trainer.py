# training/trainer.py
import logging
import time
import os

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
# ADD THESE LINES
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config.training_args import TrainingArguments
from data.data_processing import create_dataset
from model.build_model import create_model_and_tokenizer
from training.metrics import TrainingMetrics
from utils.logging_setup import setup_logging

def main():
    # --- 1. Initialization and Setup ---
    # FSDP plugin configuration. This must be done before Accelerator init.
    # REPLACE IT WITH THIS NEW BLOCK
# Create the FSDP auto-wrap policy for Llama models
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

# FSDP plugin configuration. This must be done before Accelerator init.
    # MODIFY IT TO LOOK LIKE THIS
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        auto_wrap_policy=llama_auto_wrap_policy,
        use_orig_params=True,  # This is the final, required flag
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, log_with="tensorboard", project_dir="./results/logs")
    
    setup_logging(rank=accelerator.process_index)
    logger = logging.getLogger(__name__)

    args = TrainingArguments()
    
    if accelerator.is_main_process:
        logger.info("======================================================")
        logger.info("  ProbSolSpace 1.0 Transformer Experimental Training  ")
        logger.info("======================================================")
        logger.info(f"Running with {accelerator.num_processes} GPUs using FSDP.")
        logger.info(f"Effective batch size: {args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # --- 2. Model and Tokenizer ---
    model, tokenizer = create_model_and_tokenizer(args)

    # --- 3. Optimizer and LR Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_training_steps = args.pretrain_steps + args.finetune_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    # --- 4. Training Loop ---
    def run_training_stage(stage_name, dataloader, num_steps, global_step_offset=0):
        if accelerator.is_main_process:
            logger.info(f"\n--- Starting Stage: {stage_name.upper()} for {num_steps} steps ---")

        # Prepare model and dataloader for the stage
        # The model is prepared only once if it's the first stage
        if global_step_offset == 0:
            prepared_model, prepared_optimizer, prepared_lr_scheduler = accelerator.prepare(
                model, optimizer, lr_scheduler
            )
        else:
            # For subsequent stages, components are already prepared
            prepared_model, prepared_optimizer, prepared_lr_scheduler = model, optimizer, lr_scheduler

        # REPLACE IT WITH THESE TWO LINES
        torch_dataloader = DataLoader(dataloader, batch_size=args.per_device_batch_size)
        prepared_dataloader = accelerator.prepare(torch_dataloader)
        
        # `torch.compile` is a key speedup. Apply after `accelerator.prepare`.
        # This needs to be done on all ranks.

        
        progress_bar = tqdm(
            range(num_steps), 
            disable=not accelerator.is_main_process, 
            desc=f"Stage: {stage_name}"
        )
        
        prepared_model.train()
        step_time = time.time()

        for step in range(num_steps):
            current_global_step = step + global_step_offset
            
            with accelerator.accumulate(prepared_model):
                batch = next(iter(prepared_dataloader))
                outputs = prepared_model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prepared_model.parameters(), args.grad_clip)
                
                prepared_optimizer.step()
                prepared_lr_scheduler.step()
                prepared_optimizer.zero_grad()

            # Logging and Metrics
            if (step + 1) % args.logging_steps == 0 and accelerator.is_main_process:
                # Gather loss from all processes for accurate logging
                loss_val = accelerator.gather(loss.repeat(args.per_device_batch_size)).mean().item()
                
                # Calculate metrics
                new_step_time = time.time()
                elapsed_step_time = new_step_time - step_time
                num_tokens = batch['input_ids'].numel() * accelerator.num_processes
                metrics_tracker.update(current_global_step, loss_val, num_tokens, elapsed_step_time)
                
                # Update progress bar
                progress_bar_str = metrics_tracker.get_progress_bar_str(current_global_step, loss_val)
                progress_bar.set_postfix_str(progress_bar_str)

                # Log to Tensorboard
                accelerator.log({"loss": loss_val, "perplexity": math.exp(loss_val)}, step=current_global_step)
                
                step_time = new_step_time
            
            progress_bar.update(1)
        
        progress_bar.close()
        return prepared_model, prepared_optimizer, prepared_lr_scheduler


    # --- 5. Orchestrate Stages ---
    metrics_tracker = TrainingMetrics(total_steps=total_training_steps)
    stage_boundaries = {}

    # Stage 1: Pre-training
    pretrain_dataloader = create_dataset(args, tokenizer, "pretrain")
    model, optimizer, lr_scheduler = run_training_stage("pretrain", pretrain_dataloader, args.pretrain_steps, 0)
    stage_boundaries['finetune_start'] = args.pretrain_steps

    # Stage 2: Fine-tuning
    finetune_dataloader = create_dataset(args, tokenizer, "finetune")
    run_training_stage("finetune", finetune_dataloader, args.finetune_steps, args.pretrain_steps)
    
    accelerator.wait_for_everyone()

    # --- 6. Saving Final Model and Report ---
    if accelerator.is_main_process:
        logger.info("\n--- Training Complete ---")
        logger.info(f"Saving final model to {args.final_model_dir}")
        
        # `unwrap_model` is necessary to get the actual model out of the FSDP/Compiled wrappers
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save the model and tokenizer
        unwrapped_model.save_pretrained(
            args.final_model_dir,
            state_dict=accelerator.get_state_dict(model)
        )
        tokenizer.save_pretrained(args.final_model_dir)

        # Generate and save final metrics plot
        metrics_tracker.plot_and_save(args.metrics_report_path, stage_boundaries)
        logger.info("All operations complete. ProbSolSpace 1.0 training finished.")


if __name__ == "__main__":
    main()