import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import *
import torch.nn.functional as F
import gensim.downloader as api

with open('train.txt', 'r', encoding='utf-8') as f:
    firstTrain = []
    fourthTrain = []
    fifthTrain = []
    for line in f:
        row = line.strip().split('\t')
        rowThree = eval(row[3])
        rowFour = eval(row[4])
        lowerRowThree = []
        for i in rowThree:
            for j in i:
                lowerRowThree.append(str(j).lower())
        lowerRowFour = []
        for i in rowFour:
            for j in i:
                lowerRowFour.append(str(j).lower())
        
        firstTrain.append(int(row[0]))
        fourthTrain.append(lowerRowThree)
        fifthTrain.append(lowerRowFour)  



with open('test.txt', 'r', encoding='utf-8') as f:
    firstTest = []
    fourthTest = []
    fifthTest = []
    for line in f:
        row = line.strip().split('\t')
        rowThree = eval(row[3])
        rowFour = eval(row[4])
        lowerRowThree = []
        for i in rowThree:
            for j in i:
                lowerRowThree.append(str(j).lower())
        lowerRowFour = []
        for i in rowFour:
            for j in i:
                lowerRowFour.append(str(j).lower())
        
        firstTest.append(int(row[0]))
        fourthTest.append(lowerRowThree)
        fifthTest.append(lowerRowFour) 


maxExtractedLength = 0

for i in range(0, len(fourthTrain)):
    if len(fourthTrain[i]) > maxExtractedLength:
        maxExtractedLength = len(fourthTrain[i])

for i in range(0, len(fifthTrain)):
    if len(fifthTrain[i]) > maxExtractedLength:
        maxExtractedLength = len(fifthTrain[i])

for i in range(0, len(fourthTest)):
    if len(fourthTest[i]) > maxExtractedLength:
        maxExtractedLength = len(fourthTest[i])

for i in range(0, len(fifthTest)):
    if len(fifthTest[i]) > maxExtractedLength:
        maxExtractedLength = len(fifthTest[i])


model = api.load("glove-twitter-50")


paddingVector = np.zeros(50)

paddingVector = paddingVector.astype(float)

fourthListTensorTrain = []
fifthListTensorTrain = []
newLabelTrain = []

for i in range(0, len(fourthTrain)):
    try:
        vectorsOne = [model[wordOne] for wordOne in fourthTrain[i]]
        numpyArrayOne = np.array(vectorsOne)
        numpyArrayOne = numpyArrayOne.tolist()
        for j in range(0, maxExtractedLength - len(fourthTrain[i])):
            numpyArrayOne.append(paddingVector)
        numpyArrayOne = np.array(numpyArrayOne)
        tensorOne = torch.tensor(numpyArrayOne, dtype=torch.float32)
        

        vectorsTwo = [model[wordTwo] for wordTwo in fifthTrain[i]]
        numpyArrayTwo = np.array(vectorsTwo)
        numpyArrayTwo = numpyArrayTwo.tolist()
        for j in range(0, maxExtractedLength - len(fifthTrain[i])):
            numpyArrayTwo.append(paddingVector)
        tensorTwo = torch.tensor(numpyArrayTwo, dtype=torch.float32)
        
    except:
        continue

    fourthListTensorTrain.append(tensorOne)
    fifthListTensorTrain.append(tensorTwo)
    newLabelTrain.append(float(firstTrain[i]))


fourthListTensorTest = []
fifthListTensorTest = []
newLabelTest = []

for i in range(0, len(fourthTest)):
    try:
        vectorsOne = [model[wordOne] for wordOne in fourthTest[i]]
        numpyArrayOne = np.array(vectorsOne)
        numpyArrayOne = numpyArrayOne.tolist()
        for j in range(0, maxExtractedLength - len(fourthTest[i])):
            numpyArrayOne.append(paddingVector)
        numpyArrayOne = np.array(numpyArrayOne)
        tensorOne = torch.tensor(numpyArrayOne, dtype=torch.float32)
        

        vectorsTwo = [model[wordTwo] for wordTwo in fifthTest[i]]
        numpyArrayTwo = np.array(vectorsTwo)
        numpyArrayTwo = numpyArrayTwo.tolist()
        for j in range(0, maxExtractedLength - len(fifthTest[i])):
            numpyArrayTwo.append(paddingVector)
        numpyArrayTwo = np.array(numpyArrayTwo)
        tensorTwo = torch.tensor(numpyArrayTwo, dtype=torch.float32)
        
    except:
        continue

    fourthListTensorTest.append(tensorOne)
    fifthListTensorTest.append(tensorTwo)
    newLabelTest.append(float(firstTest[i]))


x1Train = torch.stack(fourthListTensorTrain)
x2Train = torch.stack(fifthListTensorTrain)
yTrain = torch.tensor(newLabelTrain)

trainDataset = TensorDataset(x1Train, x2Train, yTrain)

x1Test = torch.stack(fourthListTensorTest)
x2Test = torch.stack(fifthListTensorTest)
yTest = torch.tensor(newLabelTest)

testDataset = TensorDataset(x1Test, x2Test, yTest)

batchSize = 64

trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

valLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False) 

inputShape = (196, 50)

filters = 1

kernelSize = 3

units = 10

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2112, 50)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.pool2(x1)
        x1 = torch.flatten(x1, 1) 
        x1 = self.fc1(x1)

        x2 = x2.unsqueeze(1)
        x2 = self.conv1(x2)
        x2 = F.relu(x2)
        x2 = self.pool1(x2)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        x2 = self.pool2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc1(x2)

        d = torch.abs(x1 - x2).sum(dim=1)

        score = torch.exp(-d)

        return score

model = CNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


nEpochs = 1

def train(model, train_loader, criterion=criterion, optimizer=optimizer, n_epoch=nEpochs):

    model.train()

    for epoch in range(n_epoch):
        runningLoss = 0
        for i, (x1, x2, y) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(x1, x2)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

    return model

model = train(model, trainLoader)

def eval_model(model, dataloader):
    
    model.eval()
    YPred = []
    YTrue = []
    for i, (x1, x2, y) in enumerate(valLoader):
        predicted = model(x1, x2)
        for element in predicted:
            YPred.append(element.item())
        for element in y:
            YTrue.append(element.item())

    return YPred, YTrue

yPred, yTrue = eval_model(model, valLoader)

for i in range(len(yPred)):
    if yPred[i] >= 0.5:
        yPred[i] = 1.0
    else:
        yPred[i] = 0.0

acc = accuracy_score(yTrue, yPred)
f1 = f1_score(yTrue, yPred)
print(acc)
print(f1)