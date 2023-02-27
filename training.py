import json
import numpy as np 
from python_util import tokenize,stemming,bow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from model import  NeuralNet
with open('data.json','r') as f:
    data = json.load(f)

tags = []
all_words = []
xy = []
for w in data['data']:
    tag = w['tag']
    tags.append(tag)
    for pattern in w['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word,tag))

ignoring_words = ['?','/','.',',','!']
all_words = [stemming(w) for w in all_words if w not in ignoring_words]

X_train = []
Y_train = []

for (pattern_sentence,tag) in xy:
    bag = bow(pattern_sentence,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


                
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data =X_train
        self.y_data = Y_train
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

batch_size=8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
lr = 0.001
epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size)
model = NeuralNet(input_size,hidden_size,output_size)

criteria = nn.CrossEntropyLoss()
optimizes = torch.optim.Adam(model.parameters(),lr = lr)

for epoch in range(epochs):
    for (words,labels) in train_loader:

        outputs = model(words)
        loss =  criteria(outputs,labels)

        optimizes.zero_grad()
        loss.backward()
        optimizes.step()
    if (epoch+1) % 100 == 0:
        print('epoch is {}/{} and loss is {}',epoch+1,epochs,loss.item())

        print('final loss is {}',loss.item())


#saving the model

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

f = 'data.pth'
torch.save(data,f)
print('saved the data')