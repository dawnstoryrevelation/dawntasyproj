# training/metrics.py
import time
import logging
import math
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TrainingMetrics:
    """A comprehensive class to track, calculate, and report on training metrics."""
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.history = {
            "step": [], "loss": [], "perplexity": [], "throughput": []
        }

    def update(self, step: int, loss: float, num_tokens: int, step_time: float):
        """Update metrics for a single step."""
        if math.isnan(loss):
            logger.warning(f"NaN loss detected at step {step}. Skipping metrics update for this step.")
            return
            
        perplexity = math.exp(loss)
        throughput = num_tokens / step_time

        self.history["step"].append(step)
        self.history["loss"].append(loss)
        self.history["perplexity"].append(perplexity)
        self.history["throughput"].append(throughput)

    def get_progress_bar_str(self, current_step: int, current_loss: float) -> str:
        """Generates a string for the TQDM progress bar."""
        elapsed_time = time.time() - self.start_time
        avg_throughput = np.mean(self.history['throughput'][-50:]) if self.history['throughput'] else 0
        
        # Estimate remaining time
        steps_remaining = self.total_steps - current_step
        eta_seconds = (elapsed_time / current_step) * steps_remaining if current_step > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        return (
            f"Loss: {current_loss:.3f} | "
            f"PPL: {math.exp(current_loss):.2f} | "
            f"Throughput: {avg_throughput:.0f} tok/s | "
            f"ETA: {eta_str}"
        )

    def plot_and_save(self, save_path: str, stage_boundaries: dict):
        """Generates and saves a plot of the training metrics."""
        logger.info(f"Generating and saving metrics report to {save_path}...")
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle('ProbSolSpace 1.0 - Training Run Report', fontsize=16)

        # Plot Loss
        axs[0].plot(self.history['step'], self.history['loss'], color='tab:blue', alpha=0.8)
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training Loss over Steps')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # Plot Perplexity
        axs[1].plot(self.history['step'], self.history['perplexity'], color='tab:red', alpha=0.8)
        axs[1].set_ylabel('Perplexity')
        axs[1].set_title('Perplexity over Steps')
        axs[1].set_yscale('log') # Perplexity can vary widely, log scale is better
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # Plot Throughput
        axs[2].plot(self.history['step'], self.history['throughput'], color='tab:green', alpha=0.8)
        axs[2].set_ylabel('Throughput (tokens/sec)')
        axs[2].set_title('System Throughput over Steps')
        axs[2].set_xlabel('Training Step')
        axs[2].grid(True, linestyle='--', alpha=0.6)

        # Add vertical lines for stage transitions
        for ax in axs:
            if stage_boundaries.get('finetune_start'):
                ax.axvline(x=stage_boundaries['finetune_start'], color='k', linestyle=':', linewidth=2, label='Finetune Start')
        
        handles, labels = axs[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Metrics report saved.")