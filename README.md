# ECG-Project
## Dataset
PhysioNet PTB-XL, a large publicly available of 12-lead electrocardiography dataset:
https://physionet.org/content/ptb-xl/1.0.1/
## Tasks
1.Age estimation - regression 

2.Cardiac arrhyth-mia diagnosis - Classification 
## Architecture - Based on architecture described at :'Automatic diagnosis of the 12-lead ECG using a deep neural network' by Ribeiro, Antônio H et-al
![image](https://user-images.githubusercontent.com/112961476/210334307-cc42f997-f1b6-4bc0-b2a7-2e346646ec68.png)

Consists of convolutional layer followed by four residual blocks with two convolutional layers per block. The output of the last block is fed into a fully connected layer. The output of each convolutional layer is rescaled using batch normalization and fed into a rectified linear activation unit (ReLU).

## Requirements:
conda env create -f requirments.yml

## Results:
### Age estimation :

![image](https://user-images.githubusercontent.com/112961476/213423666-4d6d5621-66f4-4c12-b935-a7472c1162a7.png)

![image](https://user-images.githubusercontent.com/112961476/212991609-fcd48f0f-bbb1-4847-a765-2e58552c1746.png)


### Classification :

![image](https://user-images.githubusercontent.com/112961476/213377444-e62ed994-8f18-4125-a7a8-bcde989da7d1.png)








#### References:
1. PyTorch Doc.

2. PhysioNet

3. Bm-336018 tutorials



Guided by Eran Zvuloni
