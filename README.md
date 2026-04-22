# STRIDE Final

Final locked version of the multimodal emotion recognition model used in this repo.

The final model combines:
- unimodal Transformer encoders for text, audio, and visual features
- a shared speaker-transition router that reads same-speaker and other-speaker context
- shift-aware iterative reasoning with speaker memory and dialogue memory
- a text-anchored adaptive-refusion block that uses modality confidence and context agreement

## Final Results

- MELD: `66.43` weighted F1
- IEMOCAP: `70.47` weighted F1

These are the locked target results for this final code path.

## Repo Layout

- `train.py`: training and evaluation entrypoint
- `config.py`: locked dataset-specific hyperparameters
- `dataloader.py`: dataset loader for the common ERC pickle format
- `models/encoder.py`: unimodal Transformer encoders
- `models/str.py`: main STRIDE model wiring
- `models/sair.py`: speaker-aware iterative reasoning block
- `models/fusion.py`: final adaptive-refusion fusion block
- `utils/`: losses, metrics, seeds, and checkpoint helpers

## Data

Place the dataset pickle files under `data/`:

- `meld_multimodal_features.pkl`
- `iemocap_multimodal_features.pkl`

The loader expects the common ERC pickle structure used by prior repos.

## Model Summary

### 1. Unimodal Encoding

Each modality is first projected to the shared hidden size and encoded with a Transformer encoder. Speaker embeddings are added together with positional encodings before the encoder layers.

### 2. Shared Speaker Router

The router is reused across the three modalities. For each utterance it builds:
- a same-speaker read
- an other-speaker read

Then it uses a learned gate to mix those two relation-aware signals back into the modality state.

### 3. Shift-Aware Iterative Reasoning

The routed multimodal state is passed to a dialogue reasoning block that keeps:
- a memory for each speaker
- a shared dialogue memory

At every step the current utterance is refined by comparing it with the current speaker memory and the dialogue memory, then both memories are updated.

### 4. Text-Anchored Adaptive Refusion

This is the final contribution used in the locked model.

- Text is treated as the anchor modality.
- Audio and visual provide auxiliary support only when they are confident.
- Per-modality confidence scores bias the fusion weights.
- The model computes multimodal agreement and uses it to reduce or preserve dialogue-context influence at the final fusion stage.

This design gave the final improvement over the earlier speaker-memory baseline.

## Default Training Commands

Run MELD:

```bash
python train.py --dataset meld
```

Run IEMOCAP:

```bash
python train.py --dataset iemocap
```

The default commands use the locked settings in `config.py`.

## Outputs

Training writes to:

- `checkpoints/`
- `outputs/`

By default the final config saves:

- MELD:
  - `stride_final_meld_best.pt`
  - `stride_final_meld_metrics.json`
  - `stride_final_meld_report.txt`
- IEMOCAP:
  - `stride_final_iemocap_best.pt`
  - `stride_final_iemocap_metrics.json`
  - `stride_final_iemocap_report.txt`

## Environment

Recommended:

- Python 3.9+
- CUDA-enabled PyTorch if GPU training is available

Install dependencies with:

```bash
pip install -r requirements.txt
```
