# tagging

It covers the full pipeline: SFT, DPO/GRPO, and evaluation.

Directories and what they do:
- `sft/`: SFT training and testing
  - `sft/data_loader.py`: data loading and preprocessing
  - `sft/train.py`: SFT training entry
  - `sft/test.py`: evaluation with vLLM + semantic metrics
  - `sft/sft_trainer.sh`: training script
- `dpo/`: DPO training
  - `dpo/tune_w_rlhf.py`: DPO training entry
  - `dpo/dpo_trainer.sh`: training script
- `grop_4types/`: GRPO and reward model training
  - `grop_4types/train_reward_model.py`: reward model training
  - `grop_4types/tune_with_grop.py`: GRPO training entry
  - `grop_4types/grpo_trainer.sh`: training script
- `eval/`: inference and evaluation (RAG few-shot, constrained decoding, metrics)

How to use: run SFT training, optionally run DPO/GRPO, then evaluate with the scripts in `eval/`.

## Saved models

Please place trained / saved model files under the following directories so the evaluation and downstream scripts can find them easily:

- `sft/` — put SFT checkpoints under `sft/`.
- `dpo/` — put DPO checkpoints under `dpo/`.
- `grop_4types/` — put reward model / GRPO checkpoints under `grop_4types/`.

You can link to these folders directly in the repo browser:
 - DPO models: [Google Drive](https://drive.google.com/drive/folders/1DQwAkEEZ_s3aGfqkfHa8DaNOqipHiXo7?usp=sharing)
 - SFT models: [Google Drive](https://drive.google.com/drive/folders/1xXAG3fZlxLBLYZeQGOP8eIqpusQmsmf-?usp=sharing)
 - GRPO model: [Google Drive](https://drive.google.com/drive/folders/1eThjokTamJD08VJAXcnFs97DNqY4IIxB?usp=sharing)
 - GRPO reward model: [Google Drive](https://drive.google.com/drive/folders/11aqPhZl2ltxSsOvbYQ3CuH88xgRkwxHK?usp=sharing)


