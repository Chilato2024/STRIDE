# STRIDE 

## Final Results

- MELD: `66.43` weighted F1
- IEMOCAP: `70.47` weighted F1


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
Download the preprocessed datasets from https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K, and put them into data/.
Place the dataset pickle files under `data/`:

- `meld_multimodal_features.pkl`
- `iemocap_multimodal_features.pkl`


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
