import shap
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import matplotlib.pyplot as plt
import csv
import random
import shap

class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU())
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

def load_onemetal_data(metal):
    data = pd.read_csv(f'{metal}.csv', header=None)
    return data

def merge_data(metal_list):
    final_data = load_onemetal_data(metal_list[0])
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

if __name__ == '__main__':

    all_metal_list = ['23V', '77Ir', '45Rh', '22Ti', '42Mo', '73Ta', '44Ru', '78Pt', '74W', '25Mn', '24Cr', '27Co', '75Re', '76Os', '26Fe']
    split_index = 12
    train_data = merge_data(all_metal_list[:split_index])

    x = train_data.iloc[:, :5000].values
    all_y = train_data.iloc[:, 5000:5007].values   # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
    y = all_y

    from sklearn.model_selection import train_test_split
    x_data, x_test, y_data, y_test = train_test_split(x, y, test_size=0.05, random_state=0 )
    x_data = torch.tensor(x_data,dtype=torch.float32, requires_grad=False)
    y_data = torch.tensor(y_data,dtype=torch.float32)
    x_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    my_net = torch.load(f'm_1_shap.pkl')
    my_net.eval()

    explainer = shap.DeepExplainer(my_net, x_data)

    def load_data(element,inter,index):
        testdata = load_onemetal_data(element).to_numpy()
        print('system:',element)

        inter_index = inter
        onesample = testdata[index]

        onesample = torch.tensor(onesample, dtype=torch.float32)
        input_spec = onesample[:5000].unsqueeze(0)
        sample_input = input_spec.clone().detach().requires_grad_(True)

        shap_values = explainer.shap_values(sample_input)
        print(shap_values.shape)
        output = my_net(input_spec)
        print(output[0][inter_index])

        tar_shap = abs(shap_values[0].T[inter_index])

        idx = np.argsort(tar_shap)
        idx = idx[::-1]
        top_idx = idx[:50]

        return sample_input,top_idx


