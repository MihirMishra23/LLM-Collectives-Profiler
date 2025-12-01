# LLM Collectives Profiler

Toolkit for running small TorchTitan workloads under Nsight/Torch profilers to
collect NCCL traces for the CCL-Bench project.

## Setup

1. Create the Conda environment:
   ```
   conda create --prefix $PSCRATCH/sysml-project python=3.10
   conda activate $PSCRATCH/sysml-project
   pip3 install --pre torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/nightly/cu126
   ```
2. Install the official TorchTitan repo (included in this repo):
   ```
   cd torchtitan
   pip install -e .
   pip install -r requirements.txt
   ```
3. Download the model weights:
   ```
   python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token <your token>
   ```

## Running the launcher

1. Replace the contents of `torchtitan/models/llama3/train_configs/llama3_8b.toml` with the contents of `train_llama_config_dp.toml`
2. Update `run_train.sh` to change `NGPU` from 8 to 4
3. Run 
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999
module load conda
conda activate $PSCRATCH/sysml-project
export HF_HOME=$PSCRATCH/huggingface
cd torchtitan
```

Run 10 iterations of training on the C4 dataset:
```
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_train.sh
```

The default behavior is to run with the NCCL communication backend. If you want to run with the GLOO communication backend, add the following flag to the previous command ` --comm.backend gloo`.

## Extract the metrics
cd to the root of this repo and run `python nsys_analyzer.py <profile_dir> --config <config_file>`

For me, this command looked as follows:
`python nsys_analyzer.py torchtitan/outputs/profile_trace/iteration_10/ --config torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml`

## Notes:
- On a successful run the last message should be `Process group destroyed` — this is not an error.
- If the training run is hanging on `Preparing c4 dataset from allenai/c4`, then make sure you've run `export HF_HOME=$PSCRATCH/huggingface` before.