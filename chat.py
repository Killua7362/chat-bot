import torch.nn
import torch
import random
import json
from model import NeuralNet
from python_util import bow,tokenize
import tkinter 

with open('data.json','r') as f:
    data =json.load(f)
 
f = 'data.pth'
loading = torch.load('data.pth')
input_size=loading['input_size']
hidden_size=loading['hidden_size']
output_size=loading['output_size']
all_words = loading['all_words']
tags = loading['tags']
model_state = loading['model_state']

model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Sam'
print('Let\' chat type quit to exit')

while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)

    X= bow(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _,predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]
    if prob.item()>0.75:
        for i in data['data']:
            if tag == i['tag']:
                print(f"{bot_name}: {random.choice(i['responses'])}")
    else:
        print(f"{bot_name}: I do not understand")
