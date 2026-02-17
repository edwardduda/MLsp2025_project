from dataclasses import dataclass


@dataclass
class KANLMConfig:
    """Configuration for the KAN Language Model."""

    # --- model architecture ---
    vocab_size: int = 50257
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    context_length: int = 256
    kan_hidden: int = 512
    dropout: float = 0.1

    # --- B-spline parameters ---
    num_control_points: int = 8
    spline_degree: int = 3
    spline_range_min: float = -3.0
    spline_range_max: float = 3.0

    # --- training ---
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 5
    warmup_steps: int = 500
    max_steps: int = 50_000
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_steps: int = 50
    patience: int = 5

    # --- data ---
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"

    @property
    def head_dim(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads
