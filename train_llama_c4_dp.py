import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Always prefer the editable TorchTitan under this repository over any installed copy.
PROJECT_ROOT = Path(__file__).resolve().parent
TORCHTITAN_EDITABLE = PROJECT_ROOT / "torchtitan"
if TORCHTITAN_EDITABLE.exists():
    sys.path.insert(0, str(TORCHTITAN_EDITABLE))
else:
    raise RuntimeError(
        f"Editable TorchTitan not found at {TORCHTITAN_EDITABLE}. "
        "Run `pip install -e torchtitan` inside the repo first."
    )

from torchtitan.models.llama3 import Transformer, llama3_args
from torchtitan.optim import build_optimizer  # example
from c4_dataset import get_c4_dataloader

LLAMA_VARIANT_MAP: dict[str, str] = {
    "llama-3.1-8b": "8B",
    "llama3-8b": "8B",
    "llama-3.1-70b": "70B",
    "llama3-70b": "70B",
    "llama-3.1-405b": "405B",
    "llama3-405b": "405B",
    "llama-3.1-8b-flex": "8B_flex",
    "llama3-8b-flex": "8B_flex",
    "llama-3.1-8b-varlen": "8B_varlen",
    "llama3-8b-varlen": "8B_varlen",
}


def build_llama_model(
    variant: str,
    vocab_size: int | None,
    max_seq_len: int,
) -> Transformer:
    """Build a Llama3 Transformer using TorchTitan's native configs."""
    normalized = variant.lower()
    model_key = LLAMA_VARIANT_MAP.get(normalized, normalized)
    if model_key not in llama3_args:
        raise ValueError(
            f"Unknown Llama variant '{variant}'. "
            f"Available keys: {', '.join(sorted(llama3_args.keys()))}"
        )

    override_kwargs = {"max_seq_len": max_seq_len}
    if vocab_size is not None:
        override_kwargs["vocab_size"] = vocab_size

    model_args = replace(llama3_args[model_key], **override_kwargs)
    return Transformer(model_args)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_ddp()

    # Hyperparams (tweak as needed)
    seq_len = 2048
    per_device_batch_size = 2   # global_batch = 2 * 4 = 8
    max_steps = 10

    # 1. Build model (full 8B on each GPU)
    model = build_llama_model(
        variant="llama-3.1-8b",
        vocab_size=None,   # or specific vocab
        max_seq_len=seq_len,
        # add other config args TorchTitan expects
    ).cuda(local_rank)

    # 2. Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. Optimizer (simple AdamW example)
    optimizer = build_optimizer(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        optimizer_type="adamw",
    )

    # 4. DataLoader (C4, sharded by rank)
    dataloader = get_c4_dataloader(
        rank=rank,
        world_size=world_size,
        batch_size=per_device_batch_size,
        seq_len=seq_len,
    )

    model.train()
    step = 0
    data_iter = iter(dataloader)

    while step < max_steps:
        batch = next(data_iter)
        input_ids = batch["input_ids"].cuda(local_rank, non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(local_rank, non_blocking=True)

        # Forward: standard LM next-token prediction
        logits = model(tokens=input_ids, attention_masks=attention_mask)  # [B, T, V]
        # Shift labels by one
        labels = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # triggers NCCL all-reduce on grads
        optimizer.step()

        if rank == 0:
            print(f"Step {step} | loss: {loss.item():.4f}")

        step += 1

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
