import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_onemetal_data(metal):
    data = pd.read_csv(f'{metal}.csv', header=None)
    return data

# Merge data for different metals
def merge_data(metal_list):
    final_data = load_onemetal_data(metal_list[0])
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU(True))
        self.outlayer = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.outlayer(x)

        return x

def multi_task_loss(outputs, targets):
    total_loss = 0
    for i in range(7):
        target = targets[:, i].unsqueeze(1)
        output = outputs[:, i].unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(output, target)
        total_loss += loss
    return total_loss


if __name__ == '__main__':

    all_metal_list = ['23V', '77Ir', '45Rh', '22Ti', '42Mo', '73Ta', '44Ru', '78Pt', '74W', '25Mn', '24Cr', '27Co', '75Re', '76Os', '26Fe']

    split_index = 12  # number of known systems

    train_data = merge_data(all_metal_list[:split_index])

    transfer_data = merge_data(all_metal_list[12:])
    print('transfer_metal_list',all_metal_list[12:])

    x = train_data.iloc[:, :5000].values
    all_y = train_data.iloc[:, 5000:5007].values   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
    y = all_y

    from sklearn.model_selection import train_test_split
    x_data, x_test, y_data, y_test = train_test_split(x, y, test_size=0.05, random_state=0 )
    x_data = torch.tensor(x_data,dtype=torch.float32)
    y_data = torch.tensor(y_data,dtype=torch.float32)
    x_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    batch_size = 128
    dataloader_x = DataLoader(dataset=x_data, batch_size=batch_size)
    dataloader_y = DataLoader(dataset=y_data, batch_size=batch_size)

    lr = 0.001

    my_net = Batch_Net(5000, 4096, 2048, 1024, 258, 16,7)

    optimizer = torch.optim.Adam(my_net.parameters(), lr)

    for epoch in range(100):
        my_net.train()

        for x,y in zip(dataloader_x,dataloader_y):

            optimizer.zero_grad()
            pred = my_net(x)
            loss = multi_task_loss(pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()

        print('epoch:',epoch,'  loss:',loss.item())

    torch.save(my_net, 'm_1.pkl')

    # test
    print('--------------------Test---------------------\n')
    my_net.eval()
    outputs = my_net(x_test)
    for idx in range(7):
        true = y_test[:,idx]
        out = outputs[:,idx]
        predict = []
        for i in out:
            if i >0: predict.append(1)          # sigmoid
            else:predict.append(0)

        result1 = confusion_matrix(true, predict)
        print(f'---------------{idx}---------------')   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
        print("Confusion Matrix:")
        print(result1)
        result2 = accuracy_score(true, predict)
        print("Accuracy:", result2)

    #transfer
    print('----------------------Transfer-----------------------')
    transfer_x = transfer_data.iloc[:, :5000].values
    transfer_all_y = transfer_data.iloc[:, 5000:5007].values
    transfer_x = torch.tensor(transfer_x,dtype=torch.float32)
    transfer_all_y = torch.tensor(transfer_all_y,dtype=torch.float32)

    outputs = my_net(transfer_x)
    accuracy_list = []
    for idx in range(7):
        true = transfer_all_y[:,idx]
        out = outputs[:,idx]
        predict = []
        for i in out:
            if i >0: predict.append(1)      # sigmoid
            else:predict.append(0)
        result1 = confusion_matrix(true, predict)
        print(f'---------------{idx}---------------')   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
        print("Confusion Matrix:")
        print(result1)
        result2 = accuracy_score(true, predict)
        print("Accuracy:", result2)
        accuracy_list.append(result2)
    print('average_accuracy:', sum(accuracy_list) / len(accuracy_list))
    print('all_metal_list', all_metal_list)




