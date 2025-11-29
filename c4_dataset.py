from datasets import load_dataset
from transformers import AutoTokenizer

def get_c4_dataloader(rank, world_size, batch_size, seq_len):
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # or your local path

    def tokenize_example(example):
        text = example["text"]
        tokens = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"][0], tokens["attention_mask"][0]

    # You can wrap this in an IterableDataset that shards by rank
    from torch.utils.data import IterableDataset, DataLoader

    class C4Stream(IterableDataset):
        def __iter__(self):
            # shard by rank: skip/stride by world_size
            it = dataset.__iter__()
            idx = 0
            for ex in it:
                if idx % world_size == rank:
                    input_ids, attn_mask = tokenize_example(ex)
                    yield {"input_ids": input_ids, "attention_mask": attn_mask}
                idx += 1

    ds = C4Stream()
    return DataLoader(ds, batch_size=batch_size)
