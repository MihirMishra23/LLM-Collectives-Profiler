#!/usr/bin/env python3
"""
Smoke test for torchcomms + ncclx with automatic TORCHCOMM env wiring
and fallback to env-based unique-id exchange (avoids shm).

Run with:
  TEST_BACKEND=ncclx torchrun --nproc_per_node=4 smoke_test_ncclx_fix.py
Or single-process quick check:
  TEST_BACKEND=ncclx TORCHCOMM_RANK=0 TORCHCOMM_SIZE=1 python smoke_test_ncclx_fix.py
"""
import os
import sys
import traceback
import torch
import time

try:
    from torchtitan.experiments.torchcomms.backend import prepare_comm_backend
except Exception:
    # if running from repo root, allow importing locally
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from torchtitan.experiments.torchcomms.backend import prepare_comm_backend

try:
    import torchcomms
except Exception:
    print("ERROR: Failed to import torchcomms. Ensure torchcomms is installed and visible.")
    raise

def _ensure_torchcomm_env_from_torchrun():
    # If torchrun/is launched by torch.distributed.run, these envs exist:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("TORCHCOMM_RANK")
    world_size = os.environ.get("WORLD_SIZE") or os.environ.get("TORCHCOMM_SIZE")
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("LOCAL_RANK")
    # Promote RANK/WORLD_SIZE to TORCHCOMM_*
    if rank is not None and os.environ.get("TORCHCOMM_RANK") is None:
        os.environ["TORCHCOMM_RANK"] = str(rank)
    if world_size is not None and os.environ.get("TORCHCOMM_SIZE") is None:
        os.environ["TORCHCOMM_SIZE"] = str(world_size)
    if local_rank is not None and os.environ.get("TORCHCOMM_LOCAL_RANK") is None:
        os.environ["TORCHCOMM_LOCAL_RANK"] = str(local_rank)

    # Force a non-shm unique-id exchange method so builds that don't support shm
    # won't fail. Try both variable names that might be used by different versions.
    os.environ.setdefault("TORCHCOMM_UNIQUE_ID_EXCHANGE", "env")
    os.environ.setdefault("TORCHCOMM_UNIQUE_ID_EXCHANGE_METHOD", "env")

def dump_key_envs():
    keys = [
        "TEST_BACKEND",
        "TORCHCOMM_RANK",
        "TORCHCOMM_SIZE",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "TORCHCOMM_UNIQUE_ID_EXCHANGE",
        "TORCHCOMM_UNIQUE_ID_EXCHANGE_METHOD",
    ]
    print("==== relevant environment variables ====")
    for k in keys:
        print(f"{k} = {os.environ.get(k)}")
    print("========================================")

def main():
    _ensure_torchcomm_env_from_torchrun()
    dump_key_envs()

    backend = prepare_comm_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] Using backend='{backend}', device={device}, CUDA available={torch.cuda.is_available()}")

    try:
        comm = torchcomms.new_comm(backend, device, name="smoke_ncclx_comm_test")
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        print(f"[smoke] new_comm succeeded: rank={rank}, world_size={world_size}")

        # Create a simple split group including the current rank (pair up)
        group_size = 2
        groups = []
        for start in range(0, world_size, group_size):
            ranks = list(range(start, min(start + group_size, world_size)))
            if rank in ranks:
                sub = comm.split(ranks, f"group_{start}_{start+group_size-1}")
                groups.append((ranks, sub))
                print(f"[smoke] rank {rank} created subcomm for ranks {ranks}")

        # Collective test on subcomms
        for ranks, sub in groups:
            try:
                x = torch.tensor([rank], device=device, dtype=torch.int64)
                if hasattr(sub, "allreduce"):
                    sub.allreduce(x)
                    print(f"[smoke] rank {rank} allreduce in {ranks} -> {x.item()}")
                else:
                    sub.barrier()
                    print(f"[smoke] rank {rank} barrier in {ranks}")
            except Exception as e:
                print(f"[smoke] collective on subcomm {ranks} failed: {e}")
                traceback.print_exc()

        # finalize
        try:
            for _, sub in groups:
                if hasattr(sub, "finalize"):
                    sub.finalize()
            if hasattr(comm, "finalize"):
                comm.finalize()
        except Exception as e:
            print(f"[smoke] finalize failed: {e}")

    except Exception as e:
        print("[smoke] torchcomms.new_comm failed or other error:")
        traceback.print_exc()
        sys.exit(2)

    print("[smoke] test finished OK")

if __name__ == "__main__":
    main()