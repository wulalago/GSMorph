import os
import yaml


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    return cfg


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor2array(tensor, squeeze=False):
    """
    transfer the torch tensor to numpy array
    ----------------------------
    Parameters:
        tensor: [torch tensor] tensor to be transferred
        squeeze: [Bool] option to squeeze the tensor dimensionality
    Return:
        numpy array
    ----------------------------
    """
    if squeeze:
        tensor = tensor.squeeze()
    return tensor.detach().cpu().numpy()
