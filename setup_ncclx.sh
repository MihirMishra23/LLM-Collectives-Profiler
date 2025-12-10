#!/bin/bash
# Setup NCCLX environment - FORCE custom NCCL to load first

export NCCL_HOME=$PSCRATCH/nccl-custom

# Put NCCLX FIRST in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/build/lib:${LD_LIBRARY_PATH}

# Preload custom NCCL so it overrides whatever PyTorch was built with
export LD_PRELOAD=$NCCL_HOME/build/lib/libnccl.so:${LD_PRELOAD}

# Debug level (you can keep WARN if INFO is too spammy)
export NCCL_DEBUG=INFO

echo "✓ NCCLX environment configured"
echo "NCCL library: $NCCL_HOME/build/lib"
echo "LD_LIBRARY_PATH (first entry): $(echo $LD_LIBRARY_PATH | cut -d: -f1)"
echo "LD_PRELOAD: $LD_PRELOAD"
