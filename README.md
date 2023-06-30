# ECG-Project
## Dataset
PhysioNet PTB-XL, a large publicly available of 12-lead electrocardiography dataset:
https://physionet.org/content/ptb-xl/1.0.1/
## Tasks
1. Age estimation - regression 

2. Cardiac arrhyth-mia diagnosis - Classification 
## Model1 - Based on architecture described at : 'Automatic diagnosis of the 12-lead ECG using a deep neural network' by Ribeiro , Antônio H et-al
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


### Our next goal was to Reduce the electrocardiogram time series lead system
#### Model2: We implemented architecure that was described by Han, Chuang Shi, Li at : "ML–ResNet: A novel network to detect and locate myocardial infarction using 12 leads ECG"

![image](https://github.com/RashtyShahar/ECG-Project/assets/112961476/7c20b482-7001-4a9b-b581-685ab81ba5aa)

#### We created an algorithm that extracts features from each lead separately and linearly combines the features to produce arrythmia classifications or age estimations. Using this model, we implemented Recursive Lead Elimination (RLE) algorithm. Starting from 12 leads, we train the model on N number of leads, then in validation we create N subsets of N-1 leads. A subset is chosen to be eliminated if it has the highest average AUPRC for six arrythmia classifications or the lowest MAE for age estimation, thus reducing the lead set by one. This process is repeated until only 3 leads were left. 

![image](https://github.com/RashtyShahar/ECG-Final-Project/assets/112961476/e1358bad-5892-4ce9-85f1-115fd18a6957)


![image](https://github.com/RashtyShahar/ECG-Final-Project/assets/112961476/23d44e58-c150-4443-8e7b-02b937b6832d)


![image](https://github.com/RashtyShahar/ECG-Final-Project/assets/112961476/10d120ea-a643-4c24-a082-4b4760611b9a)







#### Our next step after choseing the the reduced lead system was to implement the UResNet architecure described at :"First Steps Towards Self-Supervised Pretraining of the 12-Lead ECG; First Steps Towards Self-Supervised Pretraining of the 12-Lead ECG" , Gedon, Daniel Ribeiro, Antônio H ,Wahlström Niklas Schön, Thomas B
![image](https://github.com/RashtyShahar/ECG-Project/assets/112961476/9657e026-a874-4291-b3bb-9b0f0ad5b6c1)

#### We used this architecure as a pre-task, The reduced set was passed through the UResNet to generate an estimation of the original 12-leads set. Subsequently, the restored signal (X_hat) was forwarded through ResNet (model1) to obtain our final results :




####  References:
1. PyTorch Doc.

2. PhysioNet

3. Bm-336018 tutorials



Guided by Eran Zvuloni
