""" utils for training """
import torch


def batch_detach_cpu(x):
    """syntax honey"""
    return batch_cpu(batch_detach(x))


def batch_detach(nested_collection):
    """ move a dict of tensors to detach. 
    no op if tensors already in detach
    """
    if isinstance(nested_collection, dict):
        for k, v in nested_collection.items():
            if isinstance(v, torch.Tensor):
                nested_collection[k] = v.detach()
            if isinstance(v, dict):
                nested_collection[k] = batch_detach(v)
            elif isinstance(v, list):
                nested_collection[k] = batch_detach(v)
    if isinstance(nested_collection, list):
        for i, v in enumerate(nested_collection):
            if isinstance(v, torch.Tensor):
                nested_collection[i] = v.detach()
            elif isinstance(v, dict):
                nested_collection[i] = batch_detach(v)
            elif isinstance(v, list):
                nested_collection[i] = batch_detach(v)
    return nested_collection


def batch_cpu(nested_collection):
    """ move a dict of tensors to cpu. 
    no op if tensors already in cpu
    """
    if isinstance(nested_collection, dict):
        for k, v in nested_collection.items():
            if isinstance(v, torch.Tensor):
                nested_collection[k] = v.cpu()
            if isinstance(v, dict):
                nested_collection[k] = batch_cpu(v)
            elif isinstance(v, list):
                nested_collection[k] = batch_cpu(v)
    if isinstance(nested_collection, list):
        for i, v in enumerate(nested_collection):
            if isinstance(v, torch.Tensor):
                nested_collection[i] = v.cpu()
            elif isinstance(v, dict):
                nested_collection[i] = batch_cpu(v)
            elif isinstance(v, list):
                nested_collection[i] = batch_cpu(v)
    return nested_collection
