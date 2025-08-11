
from pathlib import Path
from omegaconf import OmegaConf


from dataclasses import dataclass, field


from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class PathsConfig:
    train_bin: Path = MISSING
    valid_bin: Path = MISSING
    model_output: Path = MISSING


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    d_model: int = 768
    d_ff: int = 2048  # floor(d_model * 8/3 / 64) * 64
    num_layers: int = 12
    num_heads: int = 12
    rope_theta: float | None = 10000.0


@dataclass
class DeepSpeedConfig:
    enabled: bool = False
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    allgather_partitions: bool = True
    allgather_bucket_size: int = 200000000  # 2e8
    reduce_bucket_size: int = 200000000     # 2e8
    contiguous_gradients: bool = True
    cpu_offload: bool = False
    overlap_comm: bool = True
    pin_memory: bool = False
    find_unused_parameters: bool = False
    force_ds_cpu_offload: bool = False
    wall_clock_breakdown: bool = False


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 128
    eval_batch_size: int = "${training.train_batch_size}"
    train_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    compile: bool = True
    eval_iterations: int | None = 1_000
    eval_interval: int = 2_000
    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 1.5e-3
    warmup_ratio: float = 0.05
    stable_ratio: float = 0.85
    decay_ratio: float = 0.10
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 20
    save_checkpoints: bool = True
    # muP params
    embeddings_scale: float = 12
    init_std: float = 2e-2
    mup_base_filter_size: int = 1024
    mup_base_hidden_size: int = 384
    # DeepSpeed config
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs():
    OmegaConf.register_new_resolver("eval", eval)
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)
    cs.store(name="base_config", node=Config)
