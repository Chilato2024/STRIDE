from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .fusion import DialogueGuidedFusion
from .sair import ShiftAwareIterativeReasoning


@dataclass
class Outputs:
    logits: torch.Tensor
    probs: torch.Tensor
    aux_logits: dict[str, torch.Tensor] | None = None
    shift_logits: torch.Tensor | None = None
    states: dict[str, torch.Tensor] | None = None


def _build_relation_masks(speaker_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    same_speaker = speaker_ids.unsqueeze(1) == speaker_ids.unsqueeze(2)
    same_mask = same_speaker & valid.unsqueeze(1) & valid.unsqueeze(2)
    other_mask = (~same_speaker) & valid.unsqueeze(1) & valid.unsqueeze(2)
    return same_mask, other_mask


def _masked_read(scores: torch.Tensor, values: torch.Tensor, relation_mask: torch.Tensor) -> torch.Tensor:
    masked_scores = scores.masked_fill(~relation_mask, float('-inf'))
    has_any = relation_mask.any(dim=-1, keepdim=True)
    masked_scores = torch.where(has_any, masked_scores, torch.zeros_like(masked_scores))
    attn = torch.softmax(masked_scores, dim=-1)
    attn = torch.where(has_any, attn, torch.zeros_like(attn))
    return torch.matmul(attn, values)


def _compute_modality_conflict(
    text_state: torch.Tensor,
    audio_state: torch.Tensor,
    visual_state: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ta_sim = F.cosine_similarity(text_state, audio_state, dim=-1, eps=1e-6)
    tv_sim = F.cosine_similarity(text_state, visual_state, dim=-1, eps=1e-6)
    av_sim = F.cosine_similarity(audio_state, visual_state, dim=-1, eps=1e-6)
    agreement = (ta_sim + tv_sim + av_sim) / 3.0
    conflict = 1.0 - (agreement.clamp(-1.0, 1.0) + 1.0) * 0.5
    return conflict.unsqueeze(-1) * mask.unsqueeze(-1)


class SharedSpeakerTransitionRouter(nn.Module):
    """Single speaker-transition router reused across all modality streams."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.self_value_proj = nn.Linear(hidden_size, hidden_size)
        self.other_value_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size * 3, hidden_size)
        self.gate_scale = nn.Parameter(torch.tensor(0.15))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, context: torch.Tensor, speaker_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        same_mask, other_mask = _build_relation_masks(speaker_ids, mask)
        query = self.query_proj(context)
        keys = self.key_proj(context)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.hidden_size)

        self_read = _masked_read(scores, self.self_value_proj(context), same_mask)
        other_read = _masked_read(scores, self.other_value_proj(context), other_mask)
        gate_input = torch.cat([context, self_read, other_read], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        routed = context + self.dropout(
            self.gate_scale * gate * self_read + self.gate_scale * (1.0 - gate) * other_read
        )
        routed = self.output_norm(routed) * mask.unsqueeze(-1)
        return routed, {
            'route_gate': gate * mask.unsqueeze(-1),
        }


class STRIDE(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_size: int,
        num_classes: int,
        num_speakers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        fusion_heads: int = 4,
        encoder_layers: int = 1,
        encoder_ff_multiplier: int = 1,
        use_adaptive_refusion: bool = False,
        use_sair_crm: bool = False,
        disable_str: bool = False,
        disable_sair: bool = False,
        anchor_scale_init: float = 1e-3,
        context_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.disable_str = disable_str
        self.disable_sair = disable_sair
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.visual_proj = nn.Linear(visual_dim, hidden_size)
        self.speaker_embeddings = nn.Embedding(num_speakers + 1, hidden_size, padding_idx=0)

        d_ff = hidden_size * encoder_ff_multiplier
        self.text_encoder = TransformerEncoder(
            hidden_size,
            d_ff,
            fusion_heads,
            encoder_layers,
            dropout,
            max_len=max_seq_len,
        )
        self.audio_encoder = TransformerEncoder(
            hidden_size,
            d_ff,
            fusion_heads,
            encoder_layers,
            dropout,
            max_len=max_seq_len,
        )
        self.visual_encoder = TransformerEncoder(
            hidden_size,
            d_ff,
            fusion_heads,
            encoder_layers,
            dropout,
            max_len=max_seq_len,
        )

        self.shared_router = SharedSpeakerTransitionRouter(hidden_size, dropout)
        self.merge_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.shift_aware = ShiftAwareIterativeReasoning(
            hidden_size,
            num_speakers,
            dropout,
            use_sair_crm=use_sair_crm,
        )
        self.fusion = DialogueGuidedFusion(
            hidden_size,
            dropout,
            use_adaptive_refusion=use_adaptive_refusion,
            anchor_scale_init=anchor_scale_init,
            context_scale_init=context_scale_init,
        )
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

        self.text_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.audio_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.visual_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.dialogue_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.all_output_layer = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> Outputs:
        valid = mask.unsqueeze(-1)
        spk_emb = self.speaker_embeddings(speaker_ids)

        text_proj = self.text_proj(text)
        audio_proj = self.audio_proj(audio)
        visual_proj = self.visual_proj(visual)

        text_enc = self.text_encoder(text_proj, mask, spk_emb)
        audio_enc = self.audio_encoder(audio_proj, mask, spk_emb)
        visual_enc = self.visual_encoder(visual_proj, mask, spk_emb)

        if self.disable_str:
            text_state = text_enc['state'] * valid
            audio_state = audio_enc['state'] * valid
            visual_state = visual_enc['state'] * valid
            zero_route = torch.zeros_like(text_state)
            text_route = {'route_gate': zero_route}
            audio_route = {'route_gate': zero_route.clone()}
            visual_route = {'route_gate': zero_route.clone()}
        else:
            text_state, text_route = self.shared_router(text_enc['state'], speaker_ids, mask)
            audio_state, audio_route = self.shared_router(audio_enc['state'], speaker_ids, mask)
            visual_state, visual_route = self.shared_router(visual_enc['state'], speaker_ids, mask)
        modality_conflict = _compute_modality_conflict(text_state, audio_state, visual_state, mask)

        merged = self.merge_proj(torch.cat([text_state, audio_state, visual_state], dim=-1)) * valid
        if self.disable_sair:
            prev_speaker = torch.cat([speaker_ids[:, :1], speaker_ids[:, :-1]], dim=1)
            speaker_change = (speaker_ids != prev_speaker).float().unsqueeze(-1)
            zero_state = torch.zeros_like(merged)
            dialogue_state = merged
            dialogue_out = {
                'refined': dialogue_state,
                'speaker_gate': zero_state,
                'dialogue_gate': zero_state,
                'speaker_memory_trace': zero_state,
                'dialogue_memory_trace': zero_state,
                'prev_speaker_state': zero_state,
                'prev_exists': mask.new_zeros(mask.shape),
                'shift_logits': None,
                'speaker_change': speaker_change,
                'modality_conflict': modality_conflict,
            }
        else:
            dialogue_out = self.shift_aware(merged, speaker_ids, mask, modality_conflict=modality_conflict)
            dialogue_state = dialogue_out['refined']

        fusion_out = self.fusion(text_state, audio_state, visual_state, dialogue_state, mask)
        fused_state = fusion_out['fused']
        guide_for_final = fusion_out.get('guide_for_final', dialogue_state)
        final_state = self.final_proj(torch.cat([fused_state, guide_for_final], dim=-1)) * valid

        logits = self.all_output_layer(final_state)
        probs = F.softmax(logits, dim=-1)
        aux_logits = {
            'text': self.text_output_layer(text_state),
            'audio': self.audio_output_layer(audio_state),
            'visual': self.visual_output_layer(visual_state),
            'dialogue': self.dialogue_output_layer(dialogue_state),
        }
        states = {
            'text_state': text_state,
            'audio_state': audio_state,
            'visual_state': visual_state,
            'dialogue_state': dialogue_state,
            'dialogue_state_for_final': guide_for_final,
            'fused_state': fused_state,
            'final_state': final_state,
            'fusion_weights': fusion_out['fusion_weights'],
            'fusion_text_anchor': fusion_out.get('text_anchor'),
            'fusion_modality_confidences': fusion_out.get('modality_confidences'),
            'fusion_context_gate': fusion_out.get('context_gate'),
            'fusion_agreement': fusion_out.get('agreement'),
            'speaker_change': dialogue_out['speaker_change'],
            'reasoning_prev_exists': dialogue_out['prev_exists'],
            'reasoning_prev_speaker_state': dialogue_out['prev_speaker_state'],
            'reasoning_speaker_gate': dialogue_out['speaker_gate'],
            'reasoning_dialogue_gate': dialogue_out['dialogue_gate'],
            'reasoning_speaker_memory': dialogue_out['speaker_memory_trace'],
            'reasoning_dialogue_memory': dialogue_out['dialogue_memory_trace'],
            'reasoning_modality_conflict': dialogue_out.get('modality_conflict'),
            'text_encoder_attention': text_enc['attention_weights'],
            'audio_encoder_attention': audio_enc['attention_weights'],
            'visual_encoder_attention': visual_enc['attention_weights'],
            'text_route_gate': text_route['route_gate'],
            'audio_route_gate': audio_route['route_gate'],
            'visual_route_gate': visual_route['route_gate'],
        }
        return Outputs(
            logits=logits,
            probs=probs,
            aux_logits=aux_logits,
            shift_logits=dialogue_out['shift_logits'],
            states=states,
        )
