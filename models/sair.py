from __future__ import annotations

import torch
from torch import nn


class ShiftAwareIterativeReasoning(nn.Module):
    """Speaker-conditioned dialogue reasoning with per-speaker and dialogue memory."""

    def __init__(
        self,
        hidden_size: int,
        num_speakers: int,
        dropout: float = 0.1,
        use_sair_crm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_speakers = num_speakers
        self.use_sair_crm = use_sair_crm
        descriptor_dim = hidden_size * 6 + 1 + int(use_sair_crm)
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.speaker_delta = nn.Linear(hidden_size * 2, hidden_size)
        self.dialogue_delta = nn.Linear(hidden_size * 2, hidden_size)
        self.speaker_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dialogue_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.speaker_update = nn.Linear(hidden_size * 2, hidden_size)
        self.dialogue_update = nn.Linear(hidden_size * 2, hidden_size)
        self.shift_classifier = nn.Linear(hidden_size * 2, 1)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.speaker_memory_norm = nn.LayerNorm(hidden_size)
        self.dialogue_memory_norm = nn.LayerNorm(hidden_size)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.shift_scale = nn.Parameter(torch.tensor(0.1))
        if use_sair_crm:
            self.crm_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(
        self,
        fused_state: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        modality_conflict: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prev_speaker = torch.cat([speaker_ids[:, :1], speaker_ids[:, :-1]], dim=1)
        speaker_change = (speaker_ids != prev_speaker).float().unsqueeze(-1)

        batch_size, seq_len, hidden_size = fused_state.shape
        speaker_memory = fused_state.new_zeros(batch_size, self.num_speakers + 1, hidden_size)
        speaker_seen = torch.zeros(batch_size, self.num_speakers + 1, dtype=torch.bool, device=fused_state.device)
        dialogue_memory = fused_state.new_zeros(batch_size, hidden_size)
        batch_indices = torch.arange(batch_size, device=fused_state.device)

        refined_states = []
        speaker_gates = []
        dialogue_gates = []
        speaker_memory_trace = []
        dialogue_memory_trace = []
        prev_speaker_states = []
        prev_exists = []
        shift_logits = []

        for step in range(seq_len):
            f_t = fused_state[:, step, :]
            current_speaker = torch.clamp(speaker_ids[:, step].long(), min=0, max=self.num_speakers)
            valid = mask[:, step].unsqueeze(-1)
            change_signal = speaker_change[:, step, :]

            speaker_prev = speaker_memory[batch_indices, current_speaker]
            has_previous = speaker_seen[batch_indices, current_speaker] & valid.squeeze(-1).bool()
            speaker_prev = speaker_prev * has_previous.unsqueeze(-1)
            dialogue_prev = dialogue_memory

            speaker_delta = f_t - speaker_prev
            dialogue_delta = f_t - dialogue_prev
            if self.use_sair_crm:
                if modality_conflict is None:
                    raise ValueError('modality_conflict must be provided when use_sair_crm=True.')
                conflict_signal = modality_conflict[:, step, :]
            else:
                conflict_signal = None
            descriptor = torch.cat(
                [
                    f_t,
                    speaker_prev,
                    dialogue_prev,
                    speaker_delta,
                    dialogue_delta,
                    speaker_delta.abs(),
                    change_signal,
                    *( [conflict_signal] if conflict_signal is not None else [] ),
                ],
                dim=-1,
            )
            shared = self.shared_trunk(descriptor)

            spk_gate = torch.sigmoid(self.speaker_gate(shared) + self.shift_scale * change_signal)
            if conflict_signal is not None:
                conflict_centered = conflict_signal - 0.5
                dlg_gate = torch.sigmoid(self.dialogue_gate(shared) + self.crm_scale * conflict_centered)
            else:
                dlg_gate = torch.sigmoid(self.dialogue_gate(shared))
            transition = spk_gate * self.speaker_delta(shared) + dlg_gate * self.dialogue_delta(shared)
            z_t = self.output_norm(f_t + self.residual_scale * transition)
            z_t = valid * z_t

            speaker_update = torch.sigmoid(self.speaker_update(shared))
            dialogue_update = torch.sigmoid(self.dialogue_update(shared))
            updated_speaker = self.speaker_memory_norm((1.0 - speaker_update) * speaker_prev + speaker_update * z_t)
            updated_dialogue = self.dialogue_memory_norm((1.0 - dialogue_update) * dialogue_prev + dialogue_update * z_t)

            current_memory = speaker_memory[batch_indices, current_speaker]
            speaker_memory[batch_indices, current_speaker] = torch.where(
                valid.bool(),
                updated_speaker,
                current_memory,
            )
            dialogue_memory = torch.where(valid.bool(), updated_dialogue, dialogue_memory)
            speaker_seen[batch_indices, current_speaker] = speaker_seen[batch_indices, current_speaker] | valid.squeeze(-1).bool()

            refined_states.append(z_t)
            speaker_gates.append(valid * spk_gate)
            dialogue_gates.append(valid * dlg_gate)
            speaker_memory_trace.append(speaker_memory[batch_indices, current_speaker].clone())
            dialogue_memory_trace.append(dialogue_memory.clone())
            prev_speaker_states.append(speaker_prev)
            prev_exists.append(has_previous.float())
            shift_logits.append(self.shift_classifier(shared).squeeze(-1))

        refined = torch.stack(refined_states, dim=1)
        return {
            'refined': refined,
            'speaker_gate': torch.stack(speaker_gates, dim=1),
            'dialogue_gate': torch.stack(dialogue_gates, dim=1),
            'speaker_memory_trace': torch.stack(speaker_memory_trace, dim=1),
            'dialogue_memory_trace': torch.stack(dialogue_memory_trace, dim=1),
            'prev_speaker_state': torch.stack(prev_speaker_states, dim=1),
            'prev_exists': torch.stack(prev_exists, dim=1),
            'shift_logits': torch.stack(shift_logits, dim=1),
            'speaker_change': speaker_change,
            'modality_conflict': modality_conflict,
        }
