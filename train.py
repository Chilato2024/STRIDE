from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CHECKPOINT_DIR, DATA_DIR, LOCKED_RUNS, OUTPUT_DIR
from dataloader import MultimodalConversationDataset
from models.str import STRIDE
from utils.io import load_checkpoint, save_checkpoint, write_json
from utils.losses import MaskedCrossEntropyLoss
from utils.metrics import build_classification_report, compute_metrics
from utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the locked STRIDE speaker-memory model.')
    parser.add_argument('--dataset', type=str.lower, choices=['iemocap', 'meld'], required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--save-path', type=Path, default=None)
    parser.add_argument('--metrics-path', type=Path, default=None)
    parser.add_argument('--report-path', type=Path, default=None)
    parser.add_argument('--adaptive-refusion', action='store_true')
    parser.add_argument('--sair-crm', action='store_true')
    parser.add_argument('--disable-str', action='store_true', help='Ablation: bypass the shared speaker transition router.')
    parser.add_argument('--disable-sair', action='store_true', help='Ablation: bypass the iterative reasoning module.')
    parser.add_argument('--aux-loss-weight', type=float, default=None)
    parser.add_argument('--shift-loss-weight', type=float, default=None)
    return parser


def dataset_file_for(name: str, data_dir: Path) -> Path:
    return data_dir / f'{name}_multimodal_features.pkl'


def split_keys(dataset: MultimodalConversationDataset, test_ratio: float, seed: int):
    official_train = dataset.official_train_keys
    official_test = dataset.official_test_keys
    if official_train is not None and official_test is not None:
        return list(official_train), list(official_test), {'type': 'official', 'test_ratio': test_ratio}

    all_keys = list(dataset.keys)
    train_keys, test_keys = train_test_split(all_keys, test_size=test_ratio, random_state=seed, shuffle=True)
    return list(train_keys), list(test_keys), {'type': 'random', 'test_ratio': test_ratio}


def build_loader(dataset: MultimodalConversationDataset, keys: list, batch_size: int, shuffle: bool, num_workers: int):
    if len(keys) == 0:
        return None
    key_to_index = {key: idx for idx, key in enumerate(dataset.keys)}
    subset = Subset(dataset, [key_to_index[key] for key in keys])
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def build_class_weights(
    dataset: MultimodalConversationDataset,
    train_keys: list,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    labels = np.asarray(dataset.flat_labels(train_keys), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        'keys': batch['keys'],
        'text': batch['text'].to(device),
        'audio': batch['audio'].to(device),
        'visual': batch['visual'].to(device),
        'speaker_ids': batch['speaker_ids'].to(device),
        'labels': batch['labels'].to(device),
        'mask': batch['mask'].to(device),
    }


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_shift_targets(labels: torch.Tensor, speaker_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = labels.shape
    num_speakers = int(speaker_ids.max().item()) if speaker_ids.numel() > 0 else 0
    speaker_label_memory = labels.new_full((batch_size, num_speakers + 1), -100)
    speaker_seen = torch.zeros(batch_size, num_speakers + 1, dtype=torch.bool, device=labels.device)
    batch_indices = torch.arange(batch_size, device=labels.device)

    targets = mask.new_zeros((batch_size, seq_len))
    valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=labels.device)

    for step in range(seq_len):
        current_speaker = torch.clamp(speaker_ids[:, step].long(), min=0, max=num_speakers)
        current_label = labels[:, step]
        current_valid = (mask[:, step] > 0) & (current_label >= 0)
        previous_label = speaker_label_memory[batch_indices, current_speaker]
        has_previous = speaker_seen[batch_indices, current_speaker] & current_valid
        targets[:, step] = torch.where(
            has_previous,
            current_label.ne(previous_label).float(),
            targets[:, step],
        )
        valid_mask[:, step] = has_previous

        stored_label = speaker_label_memory[batch_indices, current_speaker]
        speaker_label_memory[batch_indices, current_speaker] = torch.where(
            current_valid,
            current_label,
            stored_label,
        )
        speaker_seen[batch_indices, current_speaker] = speaker_seen[batch_indices, current_speaker] | current_valid

    return targets, valid_mask


def compute_total_loss(
    outputs,
    labels,
    speaker_ids,
    mask,
    criterion,
    aux_loss_weight: float,
    shift_loss_weight: float,
) -> torch.Tensor:
    loss = criterion(outputs.logits, labels, mask)
    aux_logits = getattr(outputs, 'aux_logits', None)
    if aux_logits and aux_loss_weight > 0:
        aux_losses = [criterion(logits, labels, mask) for logits in aux_logits.values()]
        loss = loss + aux_loss_weight * torch.stack(aux_losses).mean()

    shift_logits = getattr(outputs, 'shift_logits', None)
    if shift_logits is not None and shift_loss_weight > 0:
        shift_targets, shift_valid = build_shift_targets(labels, speaker_ids, mask)
        if shift_valid.any():
            shift_loss = F.binary_cross_entropy_with_logits(
                shift_logits[shift_valid],
                shift_targets[shift_valid],
            )
            loss = loss + shift_loss_weight * shift_loss
    return loss


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scheduler,
    train: bool,
    max_grad_norm: float,
    aux_loss_weight: float,
    shift_loss_weight: float,
):
    if loader is None:
        empty = {'accuracy': 0.0, 'weighted_f1': 0.0, 'macro_f1': 0.0, 'loss': 0.0}
        return empty, np.empty((0, 0)), np.empty((0,)), np.empty((0,))

    model.train(mode=train)
    all_probs, all_labels, all_masks = [], [], []
    total_loss = 0.0
    total_items = 0.0
    progress = tqdm(loader, leave=False, disable=not sys.stderr.isatty())

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(
                text=batch['text'],
                audio=batch['audio'],
                visual=batch['visual'],
                speaker_ids=batch['speaker_ids'],
                mask=batch['mask'],
            )
            loss = compute_total_loss(
                outputs,
                batch['labels'],
                batch['speaker_ids'],
                batch['mask'],
                criterion,
                aux_loss_weight,
                shift_loss_weight,
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        probs = outputs.probs.detach().cpu().numpy().reshape(-1, outputs.probs.size(-1))
        labels = batch['labels'].detach().cpu().numpy().reshape(-1)
        mask = batch['mask'].detach().cpu().numpy().reshape(-1)
        valid = mask > 0
        probs = probs[valid]
        labels = labels[valid]
        valid_count = float(valid.sum())
        total_loss += float(loss.item()) * valid_count
        total_items += valid_count
        all_probs.append(probs)
        all_labels.append(labels)
        all_masks.append(np.ones_like(labels, dtype=np.float32))
        progress.set_description(f'loss={loss.item():.4f}')

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    mask = np.concatenate(all_masks, axis=0)
    metrics = compute_metrics(probs, labels, mask)
    metrics['loss'] = total_loss / max(total_items, 1.0)
    return metrics, probs, labels, mask


def ablation_suffix(args) -> str:
    parts: list[str] = []
    if args.disable_str:
        parts.append('nostr')
    if args.disable_sair:
        parts.append('nosair')
    return '_' + '_'.join(parts) if parts else ''


def append_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f'{path.stem}{suffix}{path.suffix}')


def resolve_output_paths(args, run_config) -> None:
    suffix = ablation_suffix(args)
    if args.save_path is None:
        args.save_path = append_suffix(CHECKPOINT_DIR / run_config.checkpoint_name, suffix)
    if args.metrics_path is None:
        args.metrics_path = append_suffix(OUTPUT_DIR / run_config.metrics_name, suffix)
    if args.report_path is None:
        args.report_path = append_suffix(OUTPUT_DIR / run_config.report_name, suffix)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_config = LOCKED_RUNS[args.dataset]
    resolve_output_paths(args, run_config)
    aux_loss_weight = run_config.aux_loss_weight if args.aux_loss_weight is None else args.aux_loss_weight
    shift_loss_weight = run_config.shift_loss_weight if args.shift_loss_weight is None else args.shift_loss_weight

    set_seed(args.seed)
    if args.cpu_threads is not None and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    dataset_path = dataset_file_for(args.dataset, args.data_dir)
    dataset = MultimodalConversationDataset(
        dataset_path,
        text_feature_mode=run_config.text_feature_mode,
        text_bank_index=run_config.text_bank_index,
    )
    max_seq_len = max(run_config.max_seq_len, dataset.stats.max_seq_len)
    train_keys, test_keys, split_info = split_keys(dataset, run_config.test_ratio, args.seed)

    train_loader = build_loader(dataset, train_keys, run_config.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = build_loader(dataset, test_keys, run_config.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = dataset.stats.num_classes
    num_speakers = dataset.stats.num_speakers
    class_weights = build_class_weights(dataset, train_keys, num_classes, device) if run_config.use_class_weights else None
    criterion = MaskedCrossEntropyLoss(weight=class_weights)

    model = STRIDE(
        text_dim=dataset.stats.text_dim,
        audio_dim=dataset.stats.audio_dim,
        visual_dim=dataset.stats.visual_dim,
        hidden_size=run_config.hidden_size,
        num_classes=num_classes,
        num_speakers=num_speakers,
        max_seq_len=max_seq_len,
        dropout=run_config.dropout,
        fusion_heads=run_config.fusion_heads,
        encoder_layers=run_config.encoder_layers,
        encoder_ff_multiplier=run_config.encoder_ff_multiplier,
        use_adaptive_refusion=args.adaptive_refusion or run_config.use_adaptive_refusion,
        use_sair_crm=args.sair_crm or run_config.use_sair_crm,
        disable_str=args.disable_str,
        disable_sair=args.disable_sair,
        anchor_scale_init=run_config.anchor_scale_init,
        context_scale_init=run_config.context_scale_init,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    total_steps = max(1, run_config.epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * run_config.warmup_ratio))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    best_score = float('-inf')
    best_epoch = 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Training locked STRIDE on {args.dataset.upper()} | device={device} | params={n_params:,}')
    print(f'  data={dataset_path}')
    print(f'  split={split_info} | train={len(train_keys)} test={len(test_keys)}')
    print(f'  dims: text={dataset.stats.text_dim} audio={dataset.stats.audio_dim} visual={dataset.stats.visual_dim}')
    print(f'  classes={num_classes} speakers={num_speakers} max_seq_len={max_seq_len}')
    print(
        f'  hidden={run_config.hidden_size} batch={run_config.batch_size} dropout={run_config.dropout} '
        f'lr={run_config.lr:.2e} wd={run_config.weight_decay:.2e} heads={run_config.fusion_heads}'
    )
    print(f'  adaptive_refusion={args.adaptive_refusion or run_config.use_adaptive_refusion}')
    print(f'  sair_crm={args.sair_crm or run_config.use_sair_crm}')
    print(f'  disable_str={args.disable_str} disable_sair={args.disable_sair}')
    print(f'  aux_loss_weight={aux_loss_weight} shift_loss_weight={shift_loss_weight}')

    checkpoint_config = {
        'dataset': args.dataset,
        'data_dir': str(args.data_dir),
        'seed': args.seed,
        'locked_run': run_config.to_dict(),
        'adaptive_refusion': args.adaptive_refusion or run_config.use_adaptive_refusion,
        'sair_crm': args.sair_crm or run_config.use_sair_crm,
        'disable_str': args.disable_str,
        'disable_sair': args.disable_sair,
        'aux_loss_weight': aux_loss_weight,
        'shift_loss_weight': shift_loss_weight,
    }

    for epoch in range(1, run_config.epochs + 1):
        train_metrics, _, _, _ = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scheduler,
            True,
            run_config.max_grad_norm,
            aux_loss_weight,
            shift_loss_weight,
        )
        test_metrics_epoch, _, _, _ = run_epoch(
            model,
            test_loader,
            device,
            criterion,
            None,
            None,
            False,
            run_config.max_grad_norm,
            aux_loss_weight,
            shift_loss_weight,
        )

        current_score = test_metrics_epoch['weighted_f1']
        improved = current_score > best_score + run_config.early_stop_min_delta
        if improved:
            best_score = current_score
            best_epoch = epoch
            save_checkpoint(args.save_path, model, optimizer, epoch, best_score, checkpoint_config)
            marker = '*'
        else:
            marker = ''

        lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch:03d} | train loss {train_metrics["loss"]:.4f} | '
            f'train F1 {train_metrics["weighted_f1"]:.2f} | '
            f'test F1 {test_metrics_epoch["weighted_f1"]:.2f} {marker} | lr {lr:.2e}'
        )

    checkpoint = load_checkpoint(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f'Loaded best checkpoint from epoch {checkpoint.get("epoch", best_epoch)}.')

    test_metrics, probs, labels, mask = run_epoch(
        model,
        test_loader,
        device,
        criterion,
        None,
        None,
        False,
        run_config.max_grad_norm,
        aux_loss_weight,
        shift_loss_weight,
    )
    report = build_classification_report(probs, labels, mask, class_names=list(run_config.class_names))
    summary = {
        'dataset': args.dataset,
        'split': split_info,
        'best_epoch': best_epoch,
        'selection_metric': run_config.model_selection,
        'best_selection_weighted_f1': round(float(best_score), 4),
        'test_accuracy': round(float(test_metrics['accuracy']), 4),
        'test_weighted_f1': round(float(test_metrics['weighted_f1']), 4),
        'test_macro_f1': round(float(test_metrics['macro_f1']), 4),
        'checkpoint': str(args.save_path),
        'seed': args.seed,
        'disable_str': args.disable_str,
        'disable_sair': args.disable_sair,
        'locked_run': run_config.to_dict(),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding='utf-8')
    write_json(args.metrics_path, summary)
    print('\nFinal test classification report')
    print(report)
    print(f'Saved report to: {args.report_path}')
    print(f'Saved metrics to: {args.metrics_path}')


if __name__ == '__main__':
    main()
