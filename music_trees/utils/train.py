import torch


def batch_detach_cpu(x):
    return batch_cpu(batch_detach(x))


def batch_detach(dict_of_tensors):
    for k, v in dict_of_tensors.items():
        if isinstance(v, torch.Tensor):
            dict_of_tensors[k] = v.detach()
        if isinstance(v, dict):
            dict_of_tensors[k] = batch_detach(v)
    return dict_of_tensors


def batch_cpu(dict_of_tensors):
    for k, v in dict_of_tensors.items():
        if isinstance(v, torch.Tensor):
            dict_of_tensors[k] = v.cpu()
        if isinstance(v, dict):
            dict_of_tensors[k] = batch_cpu(v)
    return dict_of_tensors
