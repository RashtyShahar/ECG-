import numpy as np
import torch

class ECGTransform(object):
    """
    This will transform the ECG signal into a PyTorch tensor. This is the place to apply other transformations as well, e.g., normalization, etc.
    """
    def __call__(self, signal):
        # Transform the data type from double (float64) to single (float32) to match the later network weights.
        t_signal = signal.astype(np.single)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)

        return t_signal  # Make sure I am a PyTorch Tensor