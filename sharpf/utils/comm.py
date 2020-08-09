import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_batch_size(total_batch_size):
    world_size = get_world_size()
    assert (total_batch_size > 0 and total_batch_size % world_size == 0), \
        f"Total batch size ({total_batch_size}) must be divisible by the number of gpus ({world_size})."
    batch_size = total_batch_size // world_size
    return batch_size
