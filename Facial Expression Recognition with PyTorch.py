Facial Expression Recognition with PyTorch

Author: Benjamin Miller

Business Understanding

Facial Expression Recognition (FER) is an increasingly important technology with applications in healthcare, entertainment, security, and more. This project aims to develop a high-accuracy, efficient FER model using the PyTorch framework. The model will be trained on a large dataset of facial expressions to detect emotions such as happiness, sadness, fear, anger, and surprise.

The goal is to provide businesses with a tool that can improve customer experiences, enhance security measures, and increase overall operational efficiency. For example:

Healthcare: Detect early signs of depression or anxiety in patients.

Entertainment: Enhance gaming or interactive experiences.

Security: Monitor public spaces for suspicious behavior.

By automating emotion recognition, this project has the potential to transform multiple industries by providing a reliable, scalable, and accurate solution.

Data Understanding

The Face Expression Recognition dataset from Kaggle contains 28,709 labeled grayscale images of human faces, each 48×48 pixels. The dataset includes seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral.

The dataset is split into:

Training set: 24,706 images

Test set: 4,003 images

Data is stored in CSV format, with each row containing pixel values, emotion labels, and metadata like image usage and intensity. The dataset is generally balanced across emotion classes. Images were preprocessed from FER2013 to include only frontal face poses with appropriate brightness. Some low-resolution images or artifacts remain, which could affect model performance.

Overall, this dataset provides a robust foundation for training and evaluating facial expression recognition models.

Install Libraries, Packages, and Dataset

Dataset available at kaggle

https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

__results___2_0.png
Install libraries, packages and dataset

!git clone https://github.com/parth1620/Facial-Expression-Dataset.git
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install timm
!pip install --upgrade opencv-contrib-python
     

fatal: destination path 'Facial-Expression-Dataset' already exists and is not an empty directory.
Collecting git+https://github.com/albumentations-team/albumentations
  Cloning https://github.com/albumentations-team/albumentations to /tmp/pip-req-build-i61d79fi
  Running command git clone --filter=blob:none --quiet https://github.com/albumentations-team/albumentations /tmp/pip-req-build-i61d79fi
  Resolved https://github.com/albumentations-team/albumentations to commit b59030cb5a4d03ad57dca6ce4f985e0d9d8c0dd7
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: numpy>=1.24.4 in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (2.0.2)
Requirement already satisfied: scipy>=1.10.0 in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (1.15.2)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (6.0.2)
Requirement already satisfied: pydantic>=2.9.2 in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (2.11.3)
Requirement already satisfied: albucore==0.0.24 in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (0.0.24)
Requirement already satisfied: opencv-python-headless>=4.9.0.80 in /usr/local/lib/python3.11/dist-packages (from albumentations==2.0.6) (4.11.0.86)
Requirement already satisfied: stringzilla>=3.10.4 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.24->albumentations==2.0.6) (3.12.5)
Requirement already satisfied: simsimd>=5.9.2 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.24->albumentations==2.0.6) (6.2.1)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations==2.0.6) (0.7.0)
Requirement already satisfied: pydantic-core==2.33.1 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations==2.0.6) (2.33.1)
Requirement already satisfied: typing-extensions>=4.12.2 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations==2.0.6) (4.13.2)
Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations==2.0.6) (0.4.0)
Requirement already satisfied: timm in /usr/local/lib/python3.11/dist-packages (1.0.15)
Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from timm) (2.6.0+cu124)
Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (from timm) (0.21.0+cu124)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.11/dist-packages (from timm) (6.0.2)
Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.11/dist-packages (from timm) (0.30.2)
Requirement already satisfied: safetensors in /usr/local/lib/python3.11/dist-packages (from timm) (0.5.3)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (3.18.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (2025.3.2)
Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (24.2)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (2.32.3)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (4.67.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub->timm) (4.13.2)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch->timm) (3.4.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (3.1.6)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (3.2.0)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->timm) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->timm) (1.3.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision->timm) (2.0.2)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision->timm) (11.2.1)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch->timm) (3.0.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub->timm) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub->timm) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub->timm) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub->timm) (2025.4.26)
Requirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.11/dist-packages (4.11.0.86)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.11/dist-packages (from opencv-contrib-python) (2.0.2)

Imports

import numpy as np
import matplotlib.pyplot as plt
import torch
     
Configurations

Train_Path = '/content/Facial-Expression-Dataset/train'
Validation_Path = '/content/Facial-Expression-Dataset/validation'

LR = 0.01
BATCH_SIZE = 32
EPOCHS = 25
device = 'cuda'
MODEL_NAME = 'efficientnet_b0'


     
Load Dataset

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
     

Dyanmic Augmentation

train_augs = T.Compose([
T.RandomHorizontalFlip(p=0.5), # Randomly flips 50% images for better learning
T.RandomRotation(degrees=(-20, +20)), # Randomly flips rotates images by +-20 degrees
T.ToTensor() #converts PIL or numpy array to tensor format
])

valid_augs = T.Compose([
    T.ToTensor()
])
     

trainset = ImageFolder(Train_Path, train_augs)
validset = ImageFolder(Validation_Path,valid_augs)
     

print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")
     

Total no. of examples in trainset : 28821
Total no. of examples in validset : 7066


print(trainset.class_to_idx) #labels for each emotions
     

{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}


image, label = trainset[50] #50th image
plt.imshow(image.permute(1,2,0))
plt.title(label)

''' The image tensor is in the shape (C, H, W)—Channels x Height x Width. However, matplotlib expects images in the format (H, W, C)—Height x Width x Channels.

.permute(1, 2, 0) reorders the dimensions so that it's compatible with imshow.'''
     

" The image tensor is in the shape (C, H, W)—Channels x Height x Width. However, matplotlib expects images in the format (H, W, C)—Height x Width x Channels.\n\n.permute(1, 2, 0) reorders the dimensions so that it's compatible with imshow."

Load Dataset into Batches

from torch.utils.data import DataLoader
     

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)
     

print(f"Total no. of batches in trainloader : {len(trainloader)}") #no. of files / batch size
print(f"Total no. of batches in validloader : {len(validloader)}")
     

Total no. of batches in trainloader : 901
Total no. of batches in validloader : 221


for image, label in trainloader:
  break;
print(f"One image batch shape : {image.shape}") # [batch size, channels, width, height]
print(f"One label batch shape : {label.shape}")
     

One image batch shape : torch.Size([32, 3, 48, 48])
One label batch shape : torch.Size([32])

Create Model

import timm
from torch import nn
     

class FaceModel(nn.Module):

    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss

        return logits

     

model = FaceModel()
model.to(device)
     

FaceModel(
  (eff_net): EfficientNet(
    (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNormAct2d(
      32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
      (drop): Identity()
      (act): SiLU(inplace=True)
    )
    (blocks): Sequential(
      (0): Sequential(
        (0): DepthwiseSeparableConv(
          (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn1): BatchNormAct2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn2): BatchNormAct2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (1): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
          (bn2): BatchNormAct2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (bn2): BatchNormAct2d(
            144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (2): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
          (bn2): BatchNormAct2d(
            144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
          (bn2): BatchNormAct2d(
            240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (3): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
          (bn2): BatchNormAct2d(
            240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (bn2): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (bn2): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (4): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
          (bn2): BatchNormAct2d(
            480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (5): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (6): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
          (bn2): BatchNormAct2d(
            1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (aa): Identity()
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
    )
    (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn2): BatchNormAct2d(
      1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
      (drop): Identity()
      (act): SiLU(inplace=True)
    )
    (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
    (classifier): Linear(in_features=1280, out_features=7, bias=True)
  )
)

Create Train and Eval Function

from tqdm import tqdm
     

def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
     

def train_fn(model, dataloader, optimizer, current_epo):
  model.train()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc = "EPOCH" + "[TRAIN]" + str(current_epo + 1 ) + "/" + str("EPOCH"))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    logits, loss = model(images, labels)
    loss.backward()
    optimizer.step()
    total_loss +=loss.item()
    total_acc += multiclass_accuracy(logits,labels)
    tk.set_postfix({'loss': '%6f' %float(total_loss/(t+1)), 'acc' : '%6f' %float(total_acc/(t+1))})

  return total_loss/len(dataloader), total_acc/len(dataloader)

     

def eval_fn(model, dataloader, optimizer, current_epo):
  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc = "EPOCH" + "[TRAIN]" + str(current_epo + 1 ) + "/" + str("EPOCH"))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    logits, loss = model(images, labels)
    total_loss +=loss.item()
    total_acc += multiclass_accuracy(logits,labels)
    tk.set_postfix({'loss': '%6f' %float(total_loss/(t+1)), 'acc' : '%6f' %float(total_acc/(t+1))})

  return total_loss/len(dataloader), total_acc/len(dataloader)

     
Create Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
     

best_valid_loss = np.inf

for i in range(EPOCHS):
  train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
  valid_loss, valid_acc = eval_fn(model, validloader, optimizer, i)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best-weights.pt')
    print("Saved Best Weights")
    best_valid_loss = valid_loss
     

EPOCH[TRAIN]1/EPOCH: 100%|██████████| 901/901 [00:55<00:00, 16.12it/s, loss=2.153043, acc=0.228156]
EPOCH[TRAIN]1/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 38.60it/s, loss=1.643402, acc=0.361317]

Saved Best Weights

EPOCH[TRAIN]2/EPOCH: 100%|██████████| 901/901 [00:48<00:00, 18.58it/s, loss=1.524971, acc=0.402582]
EPOCH[TRAIN]2/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 36.15it/s, loss=1.376701, acc=0.446158]

Saved Best Weights

EPOCH[TRAIN]3/EPOCH: 100%|██████████| 901/901 [00:47<00:00, 19.16it/s, loss=1.367163, acc=0.473614]
EPOCH[TRAIN]3/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.00it/s, loss=1.349189, acc=0.475309]

Saved Best Weights

EPOCH[TRAIN]4/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.23it/s, loss=1.328566, acc=0.494423]
EPOCH[TRAIN]4/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.37it/s, loss=1.230975, acc=0.536797]

Saved Best Weights

EPOCH[TRAIN]5/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.61it/s, loss=1.289667, acc=0.512527]
EPOCH[TRAIN]5/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.61it/s, loss=1.298067, acc=0.502991]
EPOCH[TRAIN]6/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.67it/s, loss=1.271537, acc=0.518825]
EPOCH[TRAIN]6/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.86it/s, loss=1.260260, acc=0.527802]
EPOCH[TRAIN]7/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.58it/s, loss=1.255195, acc=0.527828]
EPOCH[TRAIN]7/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.55it/s, loss=1.253443, acc=0.537744]
EPOCH[TRAIN]8/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.26it/s, loss=1.235867, acc=0.535818]
EPOCH[TRAIN]8/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 37.88it/s, loss=1.206755, acc=0.551873]

Saved Best Weights

EPOCH[TRAIN]9/EPOCH: 100%|██████████| 901/901 [00:48<00:00, 18.68it/s, loss=1.216638, acc=0.539093]
EPOCH[TRAIN]9/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.80it/s, loss=1.260683, acc=0.528237]
EPOCH[TRAIN]10/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.64it/s, loss=1.199704, acc=0.550077]
EPOCH[TRAIN]10/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 36.11it/s, loss=1.323811, acc=0.512813]
EPOCH[TRAIN]11/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.69it/s, loss=1.184202, acc=0.554306]
EPOCH[TRAIN]11/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.81it/s, loss=1.200422, acc=0.542029]

Saved Best Weights

EPOCH[TRAIN]12/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.25it/s, loss=1.167832, acc=0.561504]
EPOCH[TRAIN]12/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 38.21it/s, loss=1.191483, acc=0.559052]

Saved Best Weights

EPOCH[TRAIN]13/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.28it/s, loss=1.152710, acc=0.566827]
EPOCH[TRAIN]13/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.44it/s, loss=1.165296, acc=0.566949]

Saved Best Weights

EPOCH[TRAIN]14/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.47it/s, loss=1.138691, acc=0.572415]
EPOCH[TRAIN]14/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.61it/s, loss=1.109569, acc=0.589541]

Saved Best Weights

EPOCH[TRAIN]15/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.55it/s, loss=1.121997, acc=0.580943]
EPOCH[TRAIN]15/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 38.17it/s, loss=1.098476, acc=0.583656]

Saved Best Weights

EPOCH[TRAIN]16/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.81it/s, loss=1.107020, acc=0.585400]
EPOCH[TRAIN]16/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 34.45it/s, loss=1.081756, acc=0.591477]

Saved Best Weights

EPOCH[TRAIN]17/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.59it/s, loss=1.099793, acc=0.589459]
EPOCH[TRAIN]17/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.75it/s, loss=1.125283, acc=0.578881]
EPOCH[TRAIN]18/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.76it/s, loss=1.082900, acc=0.592305]
EPOCH[TRAIN]18/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 40.17it/s, loss=1.136199, acc=0.575824]
EPOCH[TRAIN]19/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.79it/s, loss=1.074682, acc=0.596553]
EPOCH[TRAIN]19/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.29it/s, loss=1.045193, acc=0.613601]

Saved Best Weights

EPOCH[TRAIN]20/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.43it/s, loss=1.058763, acc=0.602728]
EPOCH[TRAIN]20/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 36.73it/s, loss=1.053101, acc=0.603746]
EPOCH[TRAIN]21/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.80it/s, loss=1.049991, acc=0.605777]
EPOCH[TRAIN]21/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 40.07it/s, loss=1.100292, acc=0.602386]
EPOCH[TRAIN]22/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.24it/s, loss=1.038712, acc=0.610978]
EPOCH[TRAIN]22/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.28it/s, loss=1.099785, acc=0.592760]
EPOCH[TRAIN]23/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.78it/s, loss=1.034970, acc=0.612909]
EPOCH[TRAIN]23/EPOCH: 100%|██████████| 221/221 [00:06<00:00, 35.11it/s, loss=1.054380, acc=0.611436]
EPOCH[TRAIN]24/EPOCH: 100%|██████████| 901/901 [00:45<00:00, 19.97it/s, loss=1.021922, acc=0.616289]
EPOCH[TRAIN]24/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 40.25it/s, loss=1.149629, acc=0.578663]
EPOCH[TRAIN]25/EPOCH: 100%|██████████| 901/901 [00:46<00:00, 19.24it/s, loss=1.010758, acc=0.621799]
EPOCH[TRAIN]25/EPOCH: 100%|██████████| 221/221 [00:05<00:00, 39.90it/s, loss=1.042050, acc=0.607510]

Saved Best Weights

Inference

def view_classify(img, ps):

    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)

    fig, (ax1, ax2) = plt.subplots(figsize=(5,9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None
     
