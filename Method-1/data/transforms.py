import numpy as np
import torch

def to_complex(data):
    # from data[256,256,2] to [256,256]complex
    data = data[:,:,0] + 1j*data[:,:,1]
    return data


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]
