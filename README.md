# tagging

It covers the full pipeline: data creation, SFT, DPO/GRPO, and evaluation.

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

How to use: prepare the data, run SFT training, optionally run DPO/GRPO, then evaluate with the scripts in `eval/`.
