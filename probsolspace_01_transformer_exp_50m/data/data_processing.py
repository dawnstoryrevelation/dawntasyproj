# data/data_processing.py
import logging
from itertools import chain
from typing import Dict, List

from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizer

# Setup logger
logger = logging.getLogger(__name__)

# --- Start of Chat Template ---
# Using a clear, unambiguous chat format.
USER_PROMPT = "<|im_start|>user\n{instruction}<|im_end|>\n"
ASSISTANT_PROMPT = "<|im_start|>assistant\n{response}<|im_end|>\n"
# --- End of Chat Template ---

def get_chat_template_format(system_prompt: str, messages: List[Dict]) -> str:
    """Formats a list of messages into a single string using the chat template."""
    # Start with the system prompt, but outside the turn-based structure.
    full_prompt = system_prompt + "\n"
    for msg in messages:
        if msg['role'] == 'user':
            full_prompt += USER_PROMPT.format(instruction=msg['content'])
        elif msg['role'] == 'assistant':
            full_prompt += ASSISTANT_PROMPT.format(response=msg['content'])
    return full_prompt

def _process_pretrain_cosmopedia(batch: Dict, system_prompt: str) -> Dict:
    """Processes a batch from the Cosmopedia dataset."""
    return {"text": [system_prompt + "\n" + text for text in batch["text"] if text]}

def _process_pretrain_numinamath(batch: Dict, system_prompt: str) -> Dict:
    """Processes a batch from the NuminaMath dataset, formatting it as a conversation."""
    formatted_texts = []
    for q, s in zip(batch["prompt"], batch["answer"]):

        if q and s:
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": s},
            ]
            formatted_texts.append(get_chat_template_format(system_prompt, messages))
    return {"text": formatted_texts}

def _process_finetune_ultrachat(batch: Dict, system_prompt: str) -> Dict:
    """Processes a batch from the UltraChat dataset."""
    return {
        "text": [
            get_chat_template_format(system_prompt, messages)
            for messages in batch["messages"] if messages
        ]
    }

def create_dataset(args, tokenizer: PreTrainedTokenizer, stage: str):
    """
    Creates and processes the dataset for a given training stage.
    Handles streaming, interleaving, processing, and tokenization.
    """
    if stage == "pretrain":
        logger.info("Loading and interleaving pre-training datasets (Cosmopedia, NuminaMath)...")
        # Streaming is critical to handle huge datasets without filling up disk/RAM
        cosmo_ds = load_dataset(
            args.pretrain_dataset_cosmopedia,
            args.pretrain_dataset_cosmopedia_subset,
            split="train",
            streaming=True
        )
        numi_ds = load_dataset(
            args.pretrain_dataset_numinamath,
            split="train",
            streaming=True
        )

        # Process each stream independently
        processed_cosmo = cosmo_ds.map(_process_pretrain_cosmopedia, batched=True, fn_kwargs={"system_prompt": args.system_prompt})
        processed_numi = numi_ds.map(_process_pretrain_numinamath, batched=True, fn_kwargs={"system_prompt": args.system_prompt})

        # Interleave the datasets to get a mix of knowledge and reasoning in each batch
        # This is more effective than sequential training
        dataset = interleave_datasets([processed_cosmo, processed_numi], probabilities=[0.7, 0.3], seed=42)

    elif stage == "finetune":
        logger.info("Loading fine-tuning dataset (UltraChat)...")
        dataset = load_dataset(args.finetune_dataset_ultrachat, split="train", streaming=True)
        dataset = dataset.map(_process_finetune_ultrachat, batched=True, fn_kwargs={"system_prompt": args.system_prompt})
    
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    # The core tokenization and grouping logic
    def tokenize_and_chunk(examples: Dict):
        # Flatten the text from the batch
        text_list = examples['text']
        
        # Tokenize
        tokenized_outputs = tokenizer(
            text_list,
            truncation=False, # We handle truncation manually by chunking
            padding=False,
            add_special_tokens=True, # Add BOS/EOS tokens
        )

        # Concatenate all tokenized texts
        concatenated_ids = list(chain(*tokenized_outputs['input_ids']))
        total_length = len(concatenated_ids)
        
        # We drop the small remainder, ensuring every sample is exactly max_seq_length
        total_length = (total_length // args.max_seq_length) * args.max_seq_length

        # Split into chunks of max_seq_length
        result = {
            "input_ids": [
                concatenated_ids[i : i + args.max_seq_length]
                for i in range(0, total_length, args.max_seq_length)
            ]
        }
        # Create labels, which are just a copy of input_ids for Causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply tokenization and chunking
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset.column_names # Remove old 'text' column
    )
    
    logger.info(f"Dataset for stage '{stage}' created successfully.")
    return tokenized_dataset