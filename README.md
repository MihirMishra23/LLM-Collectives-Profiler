# LLM Collectives Profiler

Toolkit for running small TorchTitan workloads under Nsight/Torch profilers to
collect NCCL traces for the CCL-Bench project.

## Setup

1. Create the Conda environment:
   ```
   conda create --prefix $PSCRATCH/sysml-project python=3.10
   conda activate $PSCRATCH/sysml-project
   python3 -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 -v
   ```
2. Clone the official TorchTitan repo:
   ```
   git clone https://github.com/pytorch/torchtitan
   cd torchtitan
   pip install -r requirements.txt
   ```
3. Download the model weights:
   ```
   python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token <your token>
   ```

## Running the launcher

Replace `torchtitan/models/llama3/train_configs/llama3_8b.toml` with `train_llama_config_dp.toml`

Run 10 iterations of training on the C4 dataset:
```
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_train.sh
```

## Notes:
- On a successful run the last message should be `Process group destroyed` — this is not an error.
- If the training run is hanging on `Preparing c4 dataset from allenai/c4`, then make sure you've run `export HF_HOME=$PSCRATCH/huggingface` before