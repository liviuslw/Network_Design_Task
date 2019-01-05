import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


class Dataset(data.Dataset):
    def __init__(self, params, input_file, output_file):
        self.params = params
        self.input = pd.read_csv(input_file).values
        self.output = pd.read_csv(output_file).values

    def __getitem__(self, item):
        return self.input[item], self.output[item]

    def __len__(self):
        return self.input.shape[0]

class net_Task8(torch.nn.Module): # accuracy achieves 0.70 within 1 epoch
    def __init__(self, in_num, out_num):
        super(net_Task8, self).__init__()

        self.emb = nn.Embedding(10, 8) #0-9
        self.lstm = nn.LSTM(8, 16, batch_first=True, dropout=0)
        self.gru = nn.GRU(16, 20, batch_first=True)
        self.dense1 = nn.Linear(400, 300)
        self.dense2 = nn.Linear(300, 200)

    def forward(self,x):
        x = self.emb(x) # 32,20,8
        x, _ = self.lstm(x) # 32,20,16
        x = F.relu(x)
        x, _ = self.gru(x) #32,20,20
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1) # 32,400

        x = self.dense1(x) #32,300
        x = F.relu(x)
        x = self.dense2(x) # 32,200

        x = x.reshape(x.shape[0], 20, 10) # 32, 20, 10
        return x

def accuracy(predict, output):
    predict = F.softmax(predict,dim=-1)
    predict = torch.max(predict,-1)[1]

    pre_num = predict.numpy()
    out_num = output.numpy()
    acc = np.mean(pre_num==out_num)
    return acc

if __name__=='__main__':
    pwd = os.getcwd()
    input_file = pwd + '/task8_train_input.csv'
    output_file = pwd + '/task8_train_output.csv'
    params = {'lr': 0.02,
              'epochs': 1,
              'batch_size': 32,
              'in_num': 10,
              'out_num': 10}
    dataset = Dataset(params, input_file, output_file)
    dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    net = net_Task8(params['in_num'], params['out_num'])
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epoches']):
        result = []
        for step,batch_data in enumerate(dataloader):
            net.zero_grad()
            input = batch_data[0]
            output = batch_data[1]

            predict = net(input)

            loss = 0
            predict = predict.reshape(-1, 10)
            output = output.reshape(-1)
            loss = loss_fn(predict, output)
            
            loss.backward()
            optimizer.step() # apply gradients

            temp_r = accuracy(predict, output)
            result.append(temp_r)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e, step, temp_r))

    print('final 100 step mean accuracy:', np.mean(result[-100:]))
