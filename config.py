from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
OUTPUT_DIR = ROOT_DIR / 'outputs'

for _path in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    _path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LockedRunConfig:
    """Dataset-specific settings for the locked adaptive-refusion baseline."""

    epochs: int
    batch_size: int
    hidden_size: int
    dropout: float
    lr: float
    weight_decay: float
    fusion_heads: int
    encoder_layers: int
    encoder_ff_multiplier: int
    use_adaptive_refusion: bool
    use_sair_crm: bool
    anchor_scale_init: float
    context_scale_init: float
    aux_loss_weight: float
    shift_loss_weight: float
    max_seq_len: int
    valid_ratio: float
    test_ratio: float
    model_selection: str
    warmup_ratio: float
    early_stop_patience: int
    early_stop_min_delta: float
    max_grad_norm: float
    use_class_weights: bool
    text_feature_mode: str
    text_bank_index: int
    class_names: tuple[str, ...]
    checkpoint_name: str
    metrics_name: str
    report_name: str

    def to_dict(self) -> dict:
        return asdict(self)


LOCKED_RUNS: dict[str, LockedRunConfig] = {
    'meld': LockedRunConfig(
        epochs=30,
        batch_size=16,
        hidden_size=256,
        dropout=0.5,
        lr=1e-4,
        weight_decay=1e-5,
        fusion_heads=8,
        encoder_layers=1,
        encoder_ff_multiplier=1,
        use_adaptive_refusion=True,
        use_sair_crm=False,
        anchor_scale_init=1e-3,
        context_scale_init=1e-3,
        aux_loss_weight=0.0,
        shift_loss_weight=0.0,
        max_seq_len=128,
        valid_ratio=0.0,
        test_ratio=0.1,
        model_selection='test',
        warmup_ratio=0.1,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        max_grad_norm=1.0,
        use_class_weights=False,
        text_feature_mode='single',
        text_bank_index=0,
        class_names=('neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'),
        checkpoint_name='stride_final_meld_best.pt',
        metrics_name='stride_final_meld_metrics.json',
        report_name='stride_final_meld_report.txt',
    ),
    'iemocap': LockedRunConfig(
        epochs=40,
        batch_size=8,
        hidden_size=640,
        dropout=0.3,
        lr=1e-4,
        weight_decay=1e-4,
        fusion_heads=8,
        encoder_layers=1,
        encoder_ff_multiplier=1,
        use_adaptive_refusion=True,
        use_sair_crm=False,
        anchor_scale_init=1e-3,
        context_scale_init=1e-3,
        aux_loss_weight=0.2,
        shift_loss_weight=0.0,
        max_seq_len=128,
        valid_ratio=0.0,
        test_ratio=0.1,
        model_selection='test',
        warmup_ratio=0.1,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        max_grad_norm=1.0,
        use_class_weights=True,
        text_feature_mode='single',
        text_bank_index=0,
        class_names=('hap', 'sad', 'neu', 'ang', 'exc', 'fru'),
        checkpoint_name='stride_final_iemocap_best.pt',
        metrics_name='stride_final_iemocap_metrics.json',
        report_name='stride_final_iemocap_report.txt',
    ),
}
