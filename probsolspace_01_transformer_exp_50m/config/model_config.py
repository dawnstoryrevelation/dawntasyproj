# config/model_config.py
from transformers import LlamaConfig

def get_model_config(vocab_size: int) -> LlamaConfig:
    """
    Returns the configuration for the ProbSolSpace 50M parameter model.

    The Llama 2 architecture is chosen for its efficiency and widespread support.
    Parameters are carefully selected to be near the 50M target while maintaining
    a robust configuration (e.g., head dimension is a power of 2).

    Calculation:
    - Embedding: vocab_size * hidden_size ≈ 32000 * 768 ≈ 24.6M
    - Attention (per layer): 4 * hidden_size^2 ≈ 4 * 768^2 ≈ 2.36M
    - FFN (per layer): 2 * intermediate_size * hidden_size ≈ 2 * 3072 * 768 ≈ 4.72M
    - Total per layer: 2.36M + 4.72M ≈ 7.08M
    - Total for 8 layers: 8 * 7.08M ≈ 56.6M
    - Final Layer Norm + LM Head: Marginal
    - Total ≈ 24.6M (embeddings) + 56.6M (layers) => This is closer to 80M. Let's adjust.

    Recalculating for 50M:
    - hidden_size = 768
    - num_hidden_layers = 6
    - intermediate_size = 2048
    - Total per layer: 2.36M (attn) + 2 * 2048 * 768 (ffn) = 2.36M + 3.14M = 5.5M
    - Total for 6 layers: 6 * 5.5M = 33M
    - Total params = 24.6M (embed) + 33M (layers) ≈ 57.6M. This is a perfect size.
    """
    # RECALCULATED FOR 50M with a ~152k VOCAB SIZE
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,         # Reduced to compensate for large vocab
        intermediate_size=1024,    # Standard FFN sizing
        num_hidden_layers=8,       # Kept a reasonable depth
        num_attention_heads=8,     # Must be divisible by hidden_size
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        attention_bias=False,
        attn_implementation="flash_attention_2",
    )