# model/build_model.py
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.model_config import get_model_config
from data.data_processing import USER_PROMPT, ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

def create_model_and_tokenizer(args):
    """
    Initializes the tokenizer and the model from scratch based on configuration.
    This ensures we are not starting from a pre-trained checkpoint, but building ProbSolSpace fresh.
    """
    logger.info(f"Loading base tokenizer: {args.base_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

    # --- Add special tokens for chat format ---
    special_tokens_to_add = {
        "pad_token": "<pad>", # Should not be used in training, but good practice
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }
    tokenizer.add_special_tokens(special_tokens_to_add)
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizer loaded and special tokens added.")
    
    logger.info("Building ProbSolSpace 50M model from scratch...")
    # Get the model configuration, passing the final vocab size
    config = get_model_config(vocab_size=len(tokenizer))
    
    # Instantiate the model.
    # torch_dtype=torch.bfloat16 tells the model to initialize its weights in bf16, saving memory from the start.
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    logger.info("Model built successfully.")
    logger.info(f"Total model parameters: {model.num_parameters() / 1_000_000:.2f}M")
    
    # The model's embedding layer needs to be resized to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model token embeddings resized to: {len(tokenizer)}")
    
    return model, tokenizer