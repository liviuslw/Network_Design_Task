import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


class Dataset(data.Dataset):
    def __init__(self, input_file, output_file):
        self.input = pd.read_csv(input_file).values
        self.output = pd.read_csv(output_file).values

    def __getitem__(self, item):
        return self.input[item], self.output[item]

    def __len__(self):
        return self.input.shape[0]

class net_Task8(torch.nn.Module): # accuracy achieves 0.70 within 1 epoch
    def __init__(self):
        super(net_Task8, self).__init__()

        self.emb = nn.Embedding(10, 8) #0-9
        self.gru1 = nn.GRU(8, 16, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(32, 20, batch_first=True, bidirectional=True)
        self.gru3 = nn.GRU(40, 20, batch_first=True)
        self.dense = nn.Linear(400, 200)

    def forward(self,x):
        x = self.emb(x) # 32,20,8
        x, _ = self.gru1(x) # 32,20,32
        x = F.relu(x)
        x, _ = self.gru2(x) #32,20,40
        x = F.relu(x)
        x, _ = self.gru3(x) # 32, 20, 20
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1) # 32,400

        x = self.dense(x) #32,200
        x = x.reshape(x.shape[0], 20, 10) # 32, 20, 10
        return x

def accuracy(predict, output):
    predict = F.softmax(predict, dim=-1)
    predict = torch.max(predict, dim=-1)[1]

    pre_num = predict.numpy()
    out_num = output.numpy()
    acc = np.mean(pre_num==out_num)
    return acc

if __name__=='__main__':
    cwd = os.getcwd()
    input_file = cwd + '/task8_train_input.csv'
    output_file = cwd + '/task8_train_output.csv'
    params = {'lr': 0.02,
              'epoches': 1,
              'batch_size': 32}
    dataset = Dataset(input_file, output_file)
    dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    net = net_Task8()
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epoches']):
        result = []
        for step, batch_data in enumerate(dataloader):
            net.zero_grad()
            input = batch_data[0]
            output = batch_data[1]

            predict = net(input)

            predict = predict.reshape(-1, 10)
            output = output.reshape(-1)
            loss = loss_fn(predict, output)
            
            loss.backward()
            optimizer.step() # apply gradients

            temp_acc = accuracy(predict, output)
            result.append(temp_acc)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e, step, temp_acc))

    print('Final 100 step mean accuracy:', np.mean(result[-100:]))
    
    # eval
    test_input_file = cwd + '/task8_test_input.csv'
    test_dataset = pd.read_csv(test_input_file).values
    test_dataset = torch.from_numpy(test_dataset)

    test_predict = net(test_dataset)
    labels = []
    for i in test_dataset:
        label = np.zeros(20)
        length = len(np.nonzero(i))
        no_zero = list(reversed(i[:length]))
        label[:length] = no_zero
        labels.append(label)
    labels = torch.tensor(labels)
    acc = accuracy(test_predict, labels)
    print(acc)
