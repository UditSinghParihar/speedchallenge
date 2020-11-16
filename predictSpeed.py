import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import cv2 
from sys import exit, argv
import os


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
	def __init__(self, imagesDir, speedFile):
		super(OutdoorData, self).__init__()
		self.imagesDir = imagesDir
		self.speedFile  = speedFile

	def getImageFiles(self):
		imgFiles = os.listdir(self.imagesDir)
		imgFiles = [os.path.join(self.imagesDir, img) for img in imgFiles]

		return imgFiles

	def getSpeed(self):
		f = open(self.speedFile, 'r')
		A = f.readlines()
		f.close()

		speeds = []

		for line in A:
			speeds.append(float(line))

		return speeds

	def buildDataset(self, skip=1):
		imgFiles = self.getImageFiles()
		speeds = self.getSpeed()

		# for i in range(0, 10, skip):
		for i in range(0, len(imgFiles), skip):
			print(i, imgFiles[i], speeds[i])
		
	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass


if __name__ == '__main__':
	imagesDir = argv[1]
	speedFile = argv[2]
	
	trainingDataset = OutdoorData(imagesDir, speedFile)
	trainingDataset.buildDataset(skip=1)