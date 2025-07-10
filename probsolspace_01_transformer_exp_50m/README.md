# ProbSolSpace 1.0 Transformer Experimental (50M)

## Overview

This repository contains the complete, industry-grade training implementation for the `ProbSolSpace 1.0 Transformer Experimental` model, a 50M parameter conversational AI. This is not a toy. It is a robust, heavily optimized, and production-ready system designed to run on Google Colab Enterprise with 4 x A100 GPUs.

The implementation is a direct and uncompromising response to the six foundational mandates of the ProbSolSpace initiative.

## The Mandates: From Theory to Practice

*   **Speed Mandate:** Achieved via a trifecta of SOTA optimizations:
    1.  **`Fully Sharded Data Parallel (FSDP)`**: Distributes the model, gradients, and optimizer states across all 4 A100 GPUs, maximizing parallel computation.
    2.  **`FlashAttention-2`**: Replaces the standard attention mechanism with a memory-bandwidth optimal version, directly targeting the primary bottleneck in Transformers.
    3.  **`torch.compile()`**: JIT-compiles the entire model graph into optimized kernel fusions, dramatically reducing Python overhead and accelerating per-step time.

*   **Efficiency Mandate:** Addressed through FSDP's `FULL_SHARD` strategy, which minimizes peak memory usage per GPU to an unprecedented degree. `bfloat16` mixed-precision training further halves the memory footprint. For inference, the saved model can be quantized (e.g., GPTQ, AWQ) post-training to enable execution on consumer hardware.

*   **Power/Intelligence Mandate:** Engineered through a sophisticated two-stage, multi-domain training process:
    1.  **Pre-training:** The model is simultaneously exposed to `Cosmopedia` (a massive web text corpus for world knowledge) and `NuminaMath-CoT` (a dataset of mathematical problems with chain-of-thought reasoning). This dual-stream approach builds a foundation of both factual knowledge and logical problem-solving abilities.
    2.  **Fine-tuning:** The pre-trained model is then aligned for conversation using `UltraChat`, a high-quality, multi-turn dialogue dataset.
    The identity prompt is injected into **every single training sample** to deeply embed the model's persona.

*   **Stability Mandate:** Guaranteed through a combination of:
    *   **`bfloat16` precision:** Natively supported on A100s and far more stable than `float16`, eliminating common NaN errors from underflow/overflow.
    *   **Gradient Clipping:** Prevents exploding gradients, another source of training instability.
    *   **Robust Data Processing:** Each dataset's unique structure is handled explicitly, preventing parsing errors. Streaming ensures we never run out of RAM during data loading.

*   **Sustainability & Deliverability Mandates:** The hyper-efficient training process drastically reduces total computation time and energy consumption. This script is designed to be a "one-shot" success. It is self-contained, requires no external authentication (like `wandb`), and is launched with a single command. It delivers a trained, saved model ready for download and use.

## Setup and Execution

**Environment:** Google Colab Enterprise, Machine Type `a2-highgpu-4g` (4x A100), Python 3.10.

**Step 1: Clone the repository and navigate into the project directory.**

**Step 2: Make the launch script executable.**
```bash
chmod +x run_training.sh
```

**Step 3: Launch the training.**
This single command handles everything: dependency installation, environment configuration, and launching the distributed training job.
```bash
./run_training.sh
```

Upon completion, the trained model will be saved in the `./results/final_model` directory, and a metrics report with plots will be in `./results/training_report.png`.