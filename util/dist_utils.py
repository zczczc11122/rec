import torch
import torch.distributed as dist
import torch.distributed.nn

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def all_gather(input_tensor, nprocs, local_rank):
    """
    Gathers tensor arrays of different lengths across multiple gpus

    Parameters
    ----------
        input_tensor : tensor array
        nprocs : world size
        local_rank : local_rank

    Returns
    -------
        gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(input_tensor.size()[0]).cuda(local_rank)
    all_sizes = [torch.zeros_like(local_size) for _ in range(nprocs)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    padding_size = [size_diff] + list(input_tensor.size())[1:]

    if size_diff:
        padding = torch.zeros(padding_size, dtype=input_tensor.dtype).cuda(local_rank)
        input_tensor = torch.cat((input_tensor, padding), dim=0)

    all_qs_padded = [torch.zeros_like(input_tensor) for _ in range(nprocs)]
    dist.all_gather(all_qs_padded, input_tensor)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])

    return torch.cat(all_qs, dim=0)

def gather_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    # assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # if use_horovod:
    #     assert hvd is not None, 'Please install horovod'
    #     if gather_with_grad:
    #         all_image_features = hvd.allgather(image_features)
    #         all_text_features = hvd.allgather(text_features)
    #     else:
    #         with torch.no_grad():
    #             all_image_features = hvd.allgather(image_features)
    #             all_text_features = hvd.allgather(text_features)
    #         if not local_loss:
    #             # ensure grads for local rank when all_* features don't have a gradient
    #             gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
    #             gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
    #             gathered_image_features[rank] = image_features
    #             gathered_text_features[rank] = text_features
    #             all_image_features = torch.cat(gathered_image_features, dim=0)
    #             all_text_features = torch.cat(gathered_text_features, dim=0)
    # else:

    # We gather tensors from all gpus
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)
    return all_features
