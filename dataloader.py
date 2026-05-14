# =============================================================================
# FILE: dataloader.py
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class DatasetStats:
    text_dim: int
    base_text_dim: int
    text_bank_count: int
    audio_dim: int
    visual_dim: int
    num_classes: int
    num_speakers: int
    max_seq_len: int


class MultimodalConversationDataset(Dataset):
    """Loads the common multimodal ERC pickle format used by IEMOCAP and MELD."""

    def __init__(
        self,
        data_path: str | Path,
        text_feature_mode: str = 'single',
        text_bank_index: int = 0,
    ):
        self.data_path = Path(data_path)
        self.text_feature_mode = text_feature_mode
        self.text_bank_index = text_bank_index
        if not self.data_path.exists():
            raise FileNotFoundError(
                f'Dataset file not found: {self.data_path}. Place the pickle under STRIDE_repo/data/.'
            )
        with open(self.data_path, 'rb') as f:
            try:
                payload = pickle.load(f)
            except Exception:
                f.seek(0)
                payload = pickle.load(f, encoding='latin1')
        self._unpack_payload(payload)
        self.keys = list(self.video_labels.keys())
        self._stats = self._compute_stats()

    def _unpack_payload(self, payload: Any) -> None:
        if not isinstance(payload, (list, tuple)):
            raise TypeError(f'Expected pickle payload to be list/tuple, got {type(payload)!r}')
        if len(payload) not in (12, 13):
            raise ValueError(
                f'Unsupported payload length {len(payload)}. Expected 12 or 13 items matching the common ERC format.'
            )
        self.video_ids = payload[0]
        self.video_speakers = payload[1]
        self.video_labels = payload[2]
        self.text_feature_banks = list(payload[3:7])
        self.video_text = self.text_feature_banks[0]
        self.roberta2 = self.text_feature_banks[1]
        self.roberta3 = self.text_feature_banks[2]
        self.roberta4 = self.text_feature_banks[3]
        self.video_audio = payload[7]
        self.video_visual = payload[8]
        self.video_sentence = payload[9]
        self.train_vid = payload[10] if len(payload) >= 11 else None
        self.test_vid = payload[11] if len(payload) >= 12 else None
        self.extra = payload[12] if len(payload) == 13 else None

    def _select_text_features(self, key: Any) -> tuple[np.ndarray, np.ndarray]:
        banks = np.stack(
            [self._to_array(bank[key], np.float32) for bank in self.text_feature_banks],
            axis=1,
        )
        if self.text_feature_mode == 'single':
            text = banks[:, self.text_bank_index, :]
        elif self.text_feature_mode == 'mean':
            text = banks.mean(axis=1)
        elif self.text_feature_mode == 'concat':
            text = banks.reshape(banks.shape[0], -1)
        elif self.text_feature_mode == 'mix':
            text = banks[:, self.text_bank_index, :]
        else:
            raise ValueError(f'Unsupported text_feature_mode={self.text_feature_mode!r}')
        return text.astype(np.float32, copy=False), banks.astype(np.float32, copy=False)

    @staticmethod
    def _to_array(values: Any, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(values)
        if arr.dtype == object:
            arr = np.stack([np.asarray(v) for v in values], axis=0)
        return arr.astype(dtype, copy=False)

    @staticmethod
    def _speaker_items(raw_speakers: Any) -> list[Any]:
        if isinstance(raw_speakers, np.ndarray):
            if raw_speakers.ndim == 2:
                return [row for row in raw_speakers]
            return raw_speakers.tolist()
        return list(raw_speakers)

    @classmethod
    def _speaker_ids_from_raw(cls, raw_speakers: Any) -> np.ndarray:
        items = cls._speaker_items(raw_speakers)
        if len(items) == 0:
            return np.zeros((0,), dtype=np.int64)

        first = items[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            rows = np.asarray(items, dtype=np.float32)
            if rows.ndim != 2:
                rows = np.stack([np.asarray(x, dtype=np.float32) for x in items], axis=0)
            ids = np.zeros((rows.shape[0],), dtype=np.int64)
            nonzero = np.abs(rows).sum(axis=1) > 0
            ids[nonzero] = rows[nonzero].argmax(axis=1).astype(np.int64) + 1
            return ids

        mapping: dict[Any, int] = {}
        ids: list[int] = []
        next_id = 1
        for item in items:
            key = item.item() if isinstance(item, np.generic) else item
            if key not in mapping:
                mapping[key] = next_id
                next_id += 1
            ids.append(mapping[key])
        return np.asarray(ids, dtype=np.int64)

    def _compute_stats(self) -> DatasetStats:
        first_key = self.keys[0]
        text, text_banks = self._select_text_features(first_key)
        text_dim = int(text.shape[-1])
        base_text_dim = int(text_banks.shape[-1])
        text_bank_count = int(text_banks.shape[1])
        audio_dim = int(self._to_array(self.video_audio[first_key], np.float32).shape[-1])
        visual_dim = int(self._to_array(self.video_visual[first_key], np.float32).shape[-1])
        num_classes = 0
        num_speakers = 0
        max_seq_len = 0
        for key in self.keys:
            labels = np.asarray(self.video_labels[key], dtype=np.int64)
            if labels.size:
                num_classes = max(num_classes, int(labels.max()) + 1)
            spk_ids = self._speaker_ids_from_raw(self.video_speakers[key])
            if spk_ids.size:
                num_speakers = max(num_speakers, int(spk_ids.max()))
            max_seq_len = max(max_seq_len, int(labels.shape[0]))
        return DatasetStats(
            text_dim=text_dim,
            base_text_dim=base_text_dim,
            text_bank_count=text_bank_count,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            num_classes=max(1, num_classes),
            num_speakers=max(1, num_speakers),
            max_seq_len=max(1, max_seq_len),
        )

    @property
    def stats(self) -> DatasetStats:
        return self._stats

    @property
    def official_train_keys(self) -> list[Any] | None:
        if self.train_vid is None:
            return None
        return list(self.train_vid)

    @property
    def official_test_keys(self) -> list[Any] | None:
        if self.test_vid is None:
            return None
        return list(self.test_vid)

    def flat_labels(self, keys: Iterable[Any]) -> list[int]:
        labels: list[int] = []
        for key in keys:
            labels.extend(int(x) for x in self.video_labels[key])
        return labels

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        text_arr, text_banks_arr = self._select_text_features(key)
        text = torch.as_tensor(text_arr)
        text_banks = torch.as_tensor(text_banks_arr)
        audio = torch.as_tensor(self._to_array(self.video_audio[key], np.float32))
        visual = torch.as_tensor(self._to_array(self.video_visual[key], np.float32))
        labels = torch.as_tensor(np.asarray(self.video_labels[key], dtype=np.int64))
        speaker_ids = torch.as_tensor(self._speaker_ids_from_raw(self.video_speakers[key]))
        mask = torch.ones(labels.size(0), dtype=torch.float32)
        return {
            'key': key,
            'text': text,
            'text_banks': text_banks,
            'audio': audio,
            'visual': visual,
            'speaker_ids': speaker_ids,
            'labels': labels,
            'mask': mask,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        text = pad_sequence([item['text'] for item in batch], batch_first=True)
        text_banks = pad_sequence([item['text_banks'] for item in batch], batch_first=True)
        audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
        visual = pad_sequence([item['visual'] for item in batch], batch_first=True)
        speaker_ids = pad_sequence([item['speaker_ids'] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
        mask = pad_sequence([item['mask'] for item in batch], batch_first=True, padding_value=0.0)
        keys = [item['key'] for item in batch]
        return {
            'keys': keys,
            'text': text,
            'text_banks': text_banks,
            'audio': audio,
            'visual': visual,
            'speaker_ids': speaker_ids,
            'labels': labels,
            'mask': mask,
        }
