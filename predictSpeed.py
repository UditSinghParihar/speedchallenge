import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from sys import exit, argv


class FeatureExtract(nn.Module):
	def __init__(self, outputSize):
		super(FeatureExtract, self).__init__()
		
		self.model = models.resnet18(pretrained=True)

		# self.model.fc = nn.Sequential(nn.Linear(512, 256), 
		# 	nn.ReLU(),
		# 	nn.Dropout(0.2),
		# 	nn.Linear(256, outputSize))

		self.model.fc = nn.Sequential(nn.Linear(512, outputSize))

	def forward(self, batch):
		output = self.model(batch)

		return output
		

class SpeedPredictor(nn.Module):
	def __init__(self, inputSize=256, hiddenLayerSize=100, outputSize=1):
		super(SpeedPredictor, self).__init__()
		
		self.features = FeatureExtract(inputSize)

		self.lstm = nn.LSTM(inputSize, hiddenLayerSize)

		self.linear = nn.Linear(hiddenLayerSize, outputSize)
		
		self.hiddenLayerSize = hiddenLayerSize

		self.hiddenCell = (torch.zeros(1, 1, self.hiddenLayerSize),
			torch.zeros(1, 1, self.hiddenLayerSize))

	def forward(self, seq):
		# seq : batch, channel, width, height

		feats = self.features(seq)

		lstmOut, self.hiddenCell = self.lstm(feats.view(len(seq), 1, -1), self.hiddenCell)
		
		predictions = self.linear(lstmOut.view(len(seq), -1))

		return predictions[-1], self.hiddenCell


class OutdoorData(Dataset):
	def __init__(self, arg):
		super(OutdoorData, self).__init__()
		self.arg = arg
		
	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass