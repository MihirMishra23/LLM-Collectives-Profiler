#!/bin/bash

echo "Requesting allocation..."
salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 8 --account m4999 <<'EOF'

module load conda
conda activate micro_profiling
cd $PSCRATCH/project

# Calculate MASTER_ADDR from the allocation
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== Allocation Details ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================="

# Run from login node - srun will distribute to compute nodes
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=closest \
  bash -c 'export RANK=$SLURM_PROCID; \
           export WORLD_SIZE=$SLURM_NTASKS; \
           export LOCAL_RANK=$SLURM_LOCALID; \
           python3 micro_profiling.py --backend nccl'

EOF

echo "Job completed!"