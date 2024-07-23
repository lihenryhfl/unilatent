import torch

def pad_mask(orig_mask, prefix_len=77):
    extra_zeros = torch.ones(size=(orig_mask.shape[0], prefix_len), dtype=orig_mask.dtype, device=orig_mask.device)
    return torch.cat([extra_zeros, orig_mask], axis=1)

def pad_ids(input_ids, prefix_len=77):
    extra_zeros = torch.zeros(len(input_ids), prefix_len, dtype=torch.int64, device=input_ids.device)
    return torch.cat([extra_zeros, input_ids], axis=1)