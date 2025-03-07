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

'''Embedding expert_knowledge'''

def load_onemetal_data(metal):
    data = pd.read_csv(f'{metal}.csv', header=None)
    return data

def merge_data(metal_list):
    final_data = load_onemetal_data(metal_list[0])
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

class Batch_Net(nn.Module):
    def __init__(self, spectral_dim, catal_dim, norm_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Batch_Net, self).__init__()
        self.spec_layer = nn.Sequential(nn.Linear(spectral_dim,norm_dim), nn.BatchNorm1d(norm_dim), nn.ReLU(True))
        self.catal_layer = nn.Sequential(nn.Linear(catal_dim, norm_dim), nn.BatchNorm1d(norm_dim), nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Linear(norm_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Dropout(p=0.5))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.outlayer = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, spectrum, catal_featutres):
        spectrum = self.spec_layer(spectrum)
        catal_featutres = self.catal_layer(catal_featutres)
        x = spectrum + catal_featutres
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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

    all_metal_list = ['23V', '78Pt', '77Ir', '42Mo', '27Co', '73Ta', '25Mn', '45Rh', '24Cr', '26Fe', '76Os', '22Ti', '74W', '75Re', '44Ru']

    split_index = 12  # number of known systems

    train_data = merge_data(all_metal_list[:split_index])

    transfer_data = merge_data(all_metal_list[split_index:])

    # MAGPIE
    def magpie_func(list):
        from elementembeddings.core import Embedding
        magpie = Embedding.load_data("magpie")
        catal_features = []
        for i in list:
            element = i[2:]
            one_cata_feature = []
            feature = magpie.embeddings[element]
            for ii in range(128):
                one_cata_feature.append(feature)
            catal_features = catal_features + one_cata_feature
        catal_features = np.array(catal_features)
        #catal_features = np.delete(catal_features, [3,21], 1)
        catal_features = torch.tensor(catal_features, dtype=torch.float32)
        return catal_features
    train_catal_features = magpie_func(all_metal_list[:split_index])
    tranfer_catal_features = magpie_func(all_metal_list[split_index:])

    spectral_x = train_data.iloc[:, :5000].values
    spectral_x = torch.tensor(spectral_x,dtype=torch.float32)
    all_x = torch.cat((spectral_x, train_catal_features), 1)
    all_y = train_data.iloc[:, 5000:5007].values   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO

    y = all_y

    from sklearn.model_selection import train_test_split
    x_data_all, x_test_all, y_data, y_test = train_test_split(all_x, y, test_size=0.05, random_state=0 )
    y_data = torch.tensor(y_data,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    batch_size = 128
    dataloader_x = DataLoader(dataset=x_data_all, batch_size=batch_size)
    dataloader_y = DataLoader(dataset=y_data, batch_size=batch_size)

    lr = 0.001

    my_net = Batch_Net(5000, 22, 1024, 512, 256, 64,7)

    optimizer = torch.optim.Adam(my_net.parameters(), lr)

    for epoch in range(100):
        my_net.train()

        for x,y in zip(dataloader_x,dataloader_y):

            optimizer.zero_grad()
            spec = x[:, :5000]

            catal = x[:, 5000:]

            pred = my_net(spec,catal)
            loss = multi_task_loss(pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()

        print('epoch:',epoch,'  loss:',loss.item())


    # test
    print('--------------------test---------------------')
    my_net.eval()
    outputs = my_net(x_test_all[:, :5000],x_test_all[:, 5000:])
    for idx in range(7):
        true = y_test[:,idx]
        out = outputs[:,idx]
        predict = []
        for i in out:
            if i >0: predict.append(1)
            else:predict.append(0)

        result1 = confusion_matrix(true, predict)
        print(f'---------------{idx}---------------')   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
        print("Confusion Matrix:")
        print(result1)
        result2 = accuracy_score(true, predict)
        print("Accuracy:", result2)

    #transfer
    print('----------------------transfer-----------------------')
    transfer_x = transfer_data.iloc[:, :5000].values
    transfer_all_y = transfer_data.iloc[:, 5000:5007].values   #  CHO,CHOH,CO,COH,COOH,OCH3,OCHO
    transfer_x = torch.tensor(transfer_x,dtype=torch.float32)
    transfer_all_y = torch.tensor(transfer_all_y,dtype=torch.float32)

    outputs = my_net(transfer_x,tranfer_catal_features)
    accuracy_list = []
    for idx in range(7):
        true = transfer_all_y[:,idx]
        out = outputs[:,idx]
        predict = []
        for i in out:
            if i >0: predict.append(1)
            else:predict.append(0)
        result1 = confusion_matrix(true, predict)
        print(f'---------------{idx}---------------')   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
        print("Confusion Matrix:")
        print(result1)
        result2 = accuracy_score(true, predict)
        print("Accuracy:", result2)
        accuracy_list.append(result2)
    print('average_accuracy:', sum(accuracy_list)/len(accuracy_list))
    print('all_metal_list', all_metal_list)