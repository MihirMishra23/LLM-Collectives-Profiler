# test_ncclx.py
import torch
import torchcomms
import datetime

device = torch.device("cuda:0")
print(f"[test_ncclx] Using device: {device}")

try:
    comm = torchcomms.new_comm(
        backend="ncclx",
        device=device,
        name="test_ncclx",
        timeout=datetime.timedelta(seconds=10),
    )
    print("[test_ncclx] SUCCESS! NCCLX communicator created.")
    print("  rank:", comm.get_rank())
    print("  size:", comm.get_size())
    comm.finalize()

except Exception as e:
    print("[test_ncclx] FAILED! NCCLX is NOT available.")
    print("  ERROR:", type(e).__name__, e)
