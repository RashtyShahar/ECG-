import torch.nn as nn
import torch


class ResBlk(nn.Module):
    def __init__(self, in_channels,out_channels=64,n_samples_in=1000,n_samples_out=1000):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.n_samples_out = n_samples_out
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.downsample = self.n_samples_in // self.n_samples_out

        self.direct_path1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding='same')
        )

        skip_layers=[]
        if self.downsample>1:
            skip_layers.append(nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1))
        if self.in_channels!=self.out_channels:
            skip_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1,))
        self.skip_path = nn.Sequential(*skip_layers)

        self.direct_path2 = nn.Sequential(nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2))

    def forward(self,x1,x2):
        # Compute the two paths and add the results to each other
        direct_output = self.direct_path1(x1)
        skip_output = self.skip_path(x2)
        upper_output = direct_output + skip_output
        lower_output = self.direct_path2(upper_output)
        return lower_output,upper_output


class UnResBlk(nn.Module):
    def __init__(self, in_channels, out_channels=64, n_samples_in=1000, n_samples_out=1000):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.n_samples_out = n_samples_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = self.n_samples_in // self.n_samples_out


        # self.direct_path1 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding='same'),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.ConvTranspose1d(in_channels=64, out_channels=self.out_channels, kernel_size=3)
        # )
        self.direct_path1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.ConvTranspose1d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1)
        )

        skip_layers = []
        if self.downsample > 1:
            skip_layers.append(nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1))
        if self.in_channels != self.out_channels:
            skip_layers.append(
                nn.Conv1d(in_channels=self.in_channels*2, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0,
                          dilation=1, ))
            skip_layers.append(nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1))
            # NOTE! in_channels=self.in_channels*2
        self.skip_path = nn.Sequential(*skip_layers)

        self.direct_path2 = nn.ConvTranspose1d(self.out_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x1, x2):
        # Compute the two paths and add the results to each other
        # print('x1.shape:',x1.shape)
        # print('x2.shape:',x2.shape)
        direct_output = self.direct_path1(x1)
        # print("Direct output shape:", direct_output.shape)
        skip_output = self.skip_path(x2)
        # print("Skip output shape:", skip_output.shape)

        # Add padding to the smaller tensor
        if direct_output.size(2) < skip_output.size(2):
            padding = nn.ConstantPad1d((0, skip_output.size(2) - direct_output.size(2)), 0)
            direct_output = padding(direct_output)
        elif direct_output.size(2) > skip_output.size(2):
            padding = nn.ConstantPad1d((0, direct_output.size(2) - skip_output.size(2)), 0)
            skip_output = padding(skip_output)

        upper_output = direct_output + skip_output
        lower_output = self.direct_path2(upper_output)
        return lower_output, upper_output


class UResNet(nn.Module):
    def __init__(self,input_shape, in_channels=12, out_channels=12, in_samples=5000):
        super(UResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_samples = in_samples

        # Encoder
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=16, stride=1, padding="same",
                      dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.in_samples = ((self.CNN(torch.zeros(input_shape))).shape)[2]  # 998

        self.Res1 = ResBlk(in_channels=64, out_channels=128, n_samples_in=self.in_samples, n_samples_out=1024)
        self.Res2 = ResBlk(in_channels=128, out_channels=196, n_samples_in=1024, n_samples_out=256)
        self.Res3 = ResBlk(in_channels=196,out_channels=256,n_samples_in=256,n_samples_out=64)
        self.Res4 = ResBlk(in_channels=256,out_channels=320,n_samples_in=64,n_samples_out=16)

        #for testing
        self.Res_Blocks = nn.Sequential(self.Res1,self.Res2,self.Res3,self.Res4)

        # Bottleneck layer
        self.bottleneck = nn.Conv1d(in_channels=320, out_channels=256, kernel_size=1)

        # Decoder
        self.UnRes1 = UnResBlk(256, 196,n_samples_in=1024, n_samples_out=64)
        self.UnRes2 = UnResBlk(196, 128,n_samples_in=1024, n_samples_out=256)
        self.UnRes3 = UnResBlk(128, 64,n_samples_in=1024, n_samples_out=1024)
        self.UnRes4 = UnResBlk(64, 64,n_samples_in=1024, n_samples_out=self.in_samples)

        # self.UnRes_Blocks = nn.Sequential(self.UnRes1,self.UnRes2,self.UnRes3,self.UnRes4)

        # Compute the CNN->ResBlk1-> ->ResBlk4 output size here to use as the input size for the fully-connected part.
        low, up = self.Res1(self.CNN(torch.zeros(input_shape)), self.CNN(torch.zeros(input_shape)))
        low2, up2 = self.Res2(low, up)
        low3,up3 = self.Res3(low2,up2)
        CNN_forward,_ = self.Res4(low3,up3)

    def forward(self, x):
        # Encoder
        x = self.CNN(x)
        x1, x1_skip = self.Res1(x, x)
        x2, x2_skip = self.Res2(x1, x1_skip)
        x3, x3_skip = self.Res3(x2, x2_skip)
        x4, _ = self.Res4(x3, x3_skip)

        # Bottleneck layer
        x4 = self.bottleneck(x4)

        # Decoder
        y1, _ = self.UnRes1(x4, torch.cat((x3_skip, x4), dim=1))
        y2, _ = self.UnRes2(y1, torch.cat((x2_skip, y1), dim=1))
        y3, _ = self.UnRes3(y2, torch.cat((x1_skip, y2), dim=1))
        # y4, _ = self.UnRes4(y3, torch.cat((x, y3), dim=1))

        # Output layer
        out = nn.Conv1d(in_channels=64, out_channels=self.out_channels, kernel_size=1)(y3)

        return out


class EcgModel(nn.Module):
    def __init__(self, input_shape,task):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        self.task = task
        # Define the CNN layers in a nn.Sequential.
        # Remember to use the number of input channels as the first layer input shape.
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=16, stride=1, padding="same", dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.in_samples=((self.CNN(torch.zeros(input_shape))).shape)[2] #998

        self.Res1=ResBlk(in_channels=64,out_channels=128,n_samples_in=self.in_samples,n_samples_out=512)
        self.Res2=ResBlk(in_channels=128,out_channels=196,n_samples_in=512,n_samples_out=256)
        # self.Res3=ResBlk(in_channels=196,out_channels=256,n_samples_in=256,n_samples_out=64)
        # self.Res4=ResBlk(in_channels=256,out_channels=320,n_samples_in=64,n_samples_out=16)

        #to get 4 blocks delete this and uncomment 2 lines above
        self.Res3=ResBlk(in_channels=196,out_channels=320,n_samples_in=256,n_samples_out=16)
        # self.Res2=ResBlk(in_channels=128,out_channels=320,n_samples_in=512,n_samples_out=64)


        # Compute the CNN->ResBlk1-> ->ResBlk4 output size here to use as the input size for the fully-connected part.
        low,up = self.Res1(self.CNN(torch.zeros(input_shape)),self.CNN(torch.zeros(input_shape)))
        low2,up2 = self.Res2(low,up)
        # low3,up3 = self.Res3(low2,up2)
        # CNN_forward,_ = self.Res4(low3,up3)

        #to get 4 blocks delete this and uncomment 2 lines above
        CNN_forward,_ = self.Res3(low2,up2)
        # CNN_forward,_ = self.Res2(low,up)


        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation .

        if self.task=='classification':
            self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),  # input shape is the flattened CNN output
            nn.ReLU(),
            nn.Linear(100,6)
            # nn.Sigmoid()
        )
        elif self.task == 'age estimation':
            self.FCs = nn.Sequential(
                nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),  # input shape is the flattened CNN output
                nn.ReLU(),
                nn.Linear(100, 1),  # We need 1 neuron as an output.
            )

    def forward(self, x):
        # Forward through the CNN and ResBlks by passing x, flatten and then forward through the linears.
        x = self.CNN(x)
        x,y=self.Res1(x,x)
        x,y=self.Res2(x,y)
        x,y=self.Res3(x,y)
        # x,_=self.Res4(x,y)
        features = x.view(x.size(0), -1)
        scores = self.FCs(features)
        return torch.squeeze(scores)






