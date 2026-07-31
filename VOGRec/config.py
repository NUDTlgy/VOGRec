from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 64
    state_gnn_layers: int = 2
    max_order: int = 2
    adapter_dim: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 800
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    min_state_support: int = 2
    smoothing_alpha: float = 0.5
    evidence_temperature: float = 1.0
    backoff_weight: float = 0.01
    validation_every: int = 50
    validation_switch_epoch: int | None = 400
    validation_after_every: int = 25
    full_ranking_batch_size: int = 32
