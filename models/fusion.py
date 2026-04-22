from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GuidedModalityAdapter(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size * 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.modality_proj = nn.Linear(hidden_size, hidden_size)
        self.guide_proj = nn.Linear(hidden_size, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_state: torch.Tensor, guide_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        gate = torch.sigmoid(self.gate(self.pre_norm(torch.cat([modality_state, guide_state], dim=-1))))
        guided = self.modality_proj(modality_state) + gate * self.guide_proj(guide_state)
        guided = self.out_norm(modality_state + self.dropout(guided))
        return guided * valid


class ScalarConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, modality_state: torch.Tensor, guide_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([modality_state, guide_state], dim=-1))


class DialogueGuidedFusion(nn.Module):
    """Fuse the three modality streams using the shared dialogue state as guidance."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        use_adaptive_refusion: bool = False,
        anchor_scale_init: float = 1e-3,
        context_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.use_adaptive_refusion = use_adaptive_refusion
        self.text_adapter = GuidedModalityAdapter(hidden_size, dropout)
        self.audio_adapter = GuidedModalityAdapter(hidden_size, dropout)
        self.visual_adapter = GuidedModalityAdapter(hidden_size, dropout)
        self.weight_text = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.weight_audio = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.weight_visual = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)
        self.mix_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.mix_norm = nn.LayerNorm(hidden_size)
        self.ffn = PositionwiseFFN(hidden_size, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        if use_adaptive_refusion:
            self.text_confidence = ScalarConfidenceHead(hidden_size, dropout)
            self.audio_confidence = ScalarConfidenceHead(hidden_size, dropout)
            self.visual_confidence = ScalarConfidenceHead(hidden_size, dropout)
            self.audio_to_text = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.visual_to_text = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.anchor_gate = nn.Sequential(
                nn.LayerNorm(hidden_size * 3),
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.anchor_norm = nn.LayerNorm(hidden_size)
            self.context_gate = nn.Sequential(
                nn.LayerNorm(hidden_size * 3 + 1),
                nn.Linear(hidden_size * 3 + 1, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.anchor_scale = nn.Parameter(torch.tensor(anchor_scale_init))
            self.context_scale = nn.Parameter(torch.tensor(context_scale_init))

    def forward(
        self,
        text_state: torch.Tensor,
        audio_state: torch.Tensor,
        visual_state: torch.Tensor,
        guide_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if mask.ndim != 2:
            raise ValueError(f'Expected mask to be [B, T], got {tuple(mask.shape)}.')

        valid = mask.unsqueeze(-1)
        guided_text = self.text_adapter(text_state, guide_state, mask)
        guided_audio = self.audio_adapter(audio_state, guide_state, mask)
        guided_visual = self.visual_adapter(visual_state, guide_state, mask)
        modality_confidences = None
        text_anchor = guided_text
        context_gate = None
        agreement = None

        if self.use_adaptive_refusion:
            text_conf = torch.sigmoid(self.text_confidence(guided_text, guide_state))
            audio_conf = torch.sigmoid(self.audio_confidence(guided_audio, guide_state))
            visual_conf = torch.sigmoid(self.visual_confidence(guided_visual, guide_state))
            modality_confidences = torch.stack([text_conf, audio_conf, visual_conf], dim=-2) * valid.unsqueeze(-2)

            audio_support = self.audio_to_text(torch.cat([guided_text, guided_audio], dim=-1))
            visual_support = self.visual_to_text(torch.cat([guided_text, guided_visual], dim=-1))
            auxiliary_support = audio_conf * audio_support + visual_conf * visual_support
            anchor_gate = torch.sigmoid(
                self.anchor_gate(torch.cat([guided_text, auxiliary_support, guide_state], dim=-1))
            )
            text_anchor = self.anchor_norm(guided_text + self.anchor_scale * anchor_gate * auxiliary_support) * valid
        else:
            text_conf = audio_conf = visual_conf = None

        stacked = torch.stack([text_anchor, guided_audio, guided_visual], dim=-2)
        logits = torch.stack(
            [
                self.weight_text(torch.cat([text_anchor, guide_state], dim=-1)),
                self.weight_audio(torch.cat([guided_audio, guide_state], dim=-1)),
                self.weight_visual(torch.cat([guided_visual, guide_state], dim=-1)),
            ],
            dim=-2,
        )
        if self.use_adaptive_refusion:
            logits = logits + torch.stack([text_conf, audio_conf, visual_conf], dim=-2)
        weights = self.softmax(logits)
        weighted = torch.sum(weights * stacked, dim=-2)
        fused_delta = self.mix_proj(torch.cat([text_anchor, guided_audio, guided_visual, guide_state], dim=-1))
        fused_input = weighted + fused_delta

        if self.use_adaptive_refusion:
            ta_sim = F.cosine_similarity(text_anchor, guided_audio, dim=-1, eps=1e-6).unsqueeze(-1)
            tv_sim = F.cosine_similarity(text_anchor, guided_visual, dim=-1, eps=1e-6).unsqueeze(-1)
            av_sim = F.cosine_similarity(guided_audio, guided_visual, dim=-1, eps=1e-6).unsqueeze(-1)
            agreement = (ta_sim + tv_sim + av_sim) / 3.0
            conflict = 1.0 - (agreement.clamp(-1.0, 1.0) + 1.0) * 0.5
            context_gate = torch.sigmoid(
                self.context_gate(torch.cat([weighted, text_anchor, guide_state, conflict], dim=-1))
            )

        fused = self.mix_norm(fused_input)
        fused = self.ffn_norm(fused + self.ffn(fused))
        fused = fused * valid
        weights = weights * valid.unsqueeze(-2)
        guide_for_final = guide_state
        if context_gate is not None:
            guide_for_final = guide_state * (1.0 + self.context_scale * (context_gate - 1.0)) * valid
        return {
            'fused': fused,
            'fusion_weights': weights,
            'text_anchor': text_anchor,
            'modality_confidences': modality_confidences,
            'context_gate': context_gate * valid if context_gate is not None else None,
            'agreement': agreement * valid if agreement is not None else None,
            'guide_for_final': guide_for_final,
        }
