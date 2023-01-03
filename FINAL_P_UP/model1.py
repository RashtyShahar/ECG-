import torch.nn as nn
import torch

class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Define self.direct_path by adding the layers into a nn.Sequential. Use nn.Conv1d and nn.Relu.
        # You can use padding to avoid reducing L size, to allow the skip-connection adding.
        self.direct_path = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        # Define self.skip_layers path.
        # You should use convolution layer with a kernel size of 1 to consider the case where the input and output shapes mismatch.

        skip_layers = []
        if in_channels != 64:
            skip_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1,
                          bias=False
            ))

        self.skip_path = nn.Sequential(*skip_layers)

    def forward(self, x):

        # Compute the two paths and add the results to each other, then use ReLU (torch.relu) to activate the output.
        direct_output = self.direct_path(x)
        skip_output = self.skip_path(x)
        activated_output = torch.relu(direct_output + skip_output)

        return activated_output


class ecgNet(nn.Module):
    def __init__(self, input_shape,task):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        self.task = task

        # ------Your code------#
        # Define the CNN layers in a nn.Sequential.
        # Remember to use the number of input channels as the first layer input shape.
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            Residual(in_channels=64)
        )

        # Compute the CNN output size here to use as the input size for the fully-connected part.
        CNN_forward = self.CNN(torch.zeros(input_shape))

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        if self.task=='classification':
            self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),  # input shape is the flattened CNN output
            nn.ReLU(),
            nn.Linear(100,6),
            nn.Sigmoid()
        )
        elif self.task == 'age estimation':
            self.FCs = nn.Sequential(
                nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),  # input shape is the flattened CNN output
                nn.ReLU(),
                nn.Linear(100, 1),  # We need 1 neuron as an output.
            )
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Forward through the CNN by passing x, flatten and then forward through the linears.
        features = self.CNN(x)
        features = features.view(features.size(0), -1)
        scores = self.FCs(features)
        # ------^^^^^^^^^------#
        return torch.squeeze(scores)

