import torch.nn as nn
import torch
import torch.nn.functional as F

class Resnet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=17, padding='same'),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=11, padding='same'),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(self.out_channels)
        )

        self.skip_path = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(self.out_channels)
        )

        self.pooling=nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(5)
        )

    def forward(self,x):
        # Compute direct and skip paths at the block and sum the results
        # print('Input shape:', x.shape)
        block = self.block(x)
        # print('block shape:', block.shape)

        skip=self.skip_path(x)
        # print('skip shape:', skip.shape)
        output=self.pooling(block+skip)

        return output

class InverseResnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=11, padding='same'),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=17, padding='same'),
            nn.BatchNorm1d(self.out_channels)
        )

        self.skip_path = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(self.out_channels)
        )

        self.pooling = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=5, mode='nearest')
        )

    def forward(self, x):
        block = self.block(x)
        skip = self.skip_path(x)
        output = self.pooling(block + skip)

        return output

class UNetDecoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            InverseResnet(in_channels=in_channels, out_channels=4),
            InverseResnet(in_channels=4, out_channels=2),
            InverseResnet(in_channels=2, out_channels=1)
        )
    def forward(self, x):
        return self.decoder(x)

class RLE(nn.Module):
    def __init__(self, input_shape,task,lead_idx=None,to_test=False):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        self.task = task
        self.to_test = to_test
        self.lead_idx = lead_idx

        #initializing 12 seprate models
        self.leads = nn.ModuleList([self.create_lead() for _ in range(12)])
        # Add an instance of the UNet class with 12 leads output
        #calculte shape to get into the FC
        CNN_forward = [self.leads[i](torch.zeros(input_shape)[:,0:1,:]) for i in range(12)]
        # CNN_forward = self.get_concatenations_except_one(CNN_forward)[0]
        CNN_forward = torch.cat([inner_tensor for inner_tensor in CNN_forward], dim=0)
        # self.unet = UNet(input_shape[1], 12)
        self.unet = UNetDecoder(CNN_forward.shape[1] * CNN_forward.shape[2])

        # print(CNN_forward.shape)

        if self.task=='classification':
            self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 12),  # input shape is the flattened CNN output
            nn.ReLU(),
            nn.Linear(12,6)
            # nn.Sigmoid()
        )
        elif self.task == 'age estimation':
            self.FCs = nn.Sequential(
                nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),  # input shape is the flattened CNN output
                nn.ReLU(),
                nn.Linear(100, 1),  # We need 1 neuron as an output.
            )

    def create_lead(self):
        return nn.Sequential(
            Resnet(in_channels=1, out_channels=2),
            Resnet(in_channels=2, out_channels=4),
            Resnet(in_channels=4, out_channels=8),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
    def sum_except_one(self,lst):
        #get all combinations
        result = []
        for i in range(len(lst)):
            s = sum(lst[:i] + lst[i + 1:])
            result.append(s)
        return result

    def get_concatenations_except_one(self,tensor_list):
        results = []
        for i in range(len(tensor_list)):
            # excluded_tensor = tensor_list[i]
            included_tensors = tensor_list[:i] + tensor_list[i + 1:]
            concatenated = torch.cat(included_tensors, dim=1)
            results.append(concatenated)
        return results

    def drop_lead(self, tensor, lead_idx):
        if lead_idx is None:
            return tensor
        # drop lead at lead_idx
        else:
            new_tensor = tensor[:, :lead_idx, :]
            new_tensor = torch.cat((new_tensor, tensor[:, lead_idx + 1:, :]), dim=1)
            return new_tensor

    def forward(self, x):
        # run each lead separately through his own model , x.shape[1] is the number of leads
        single_lead_outputs_list=[self.leads[i](x[:,i:i+1,:]) for i in range(x.shape[1])]
        # inverse = [self.unet(lead) for lead in single_lead_outputs_list]

        # concatenate all leads outputs
        # output = torch.cat([inner_tensor for inner_tensor in single_lead_outputs_list], dim=0)

        #sum the all leads outputs
        single_lead_outputs_stacked = torch.stack(single_lead_outputs_list,dim=0)
        output = torch.sum(single_lead_outputs_stacked,dim=0)
        # flatten
        output = output.view(output.size(0), -1)
        # run the flatten tensor through the FC to get output
        output1 = self.FCs(output)
        # run the flatten tensor through Unet to get 12 leads ECG
        # reconstructed_ecg = self.unet(output.unsqueeze(1))

        return output1#,reconstructed_ecg





class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x1 = self.encoder(x)
        # x2 = self.middle(x)
        x3 = self.decoder(x)
        return x3











