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

'''test_solvent'''

from model import Batch_Net

def load_onemetal_data(metal,solvent):
    data = pd.read_csv(f'{metal}/{solvent}/data.csv', header=None)
    return data

# Merge data for different metals
def merge_data(metal_list,solvent):
    final_data = load_onemetal_data(metal_list[0],solvent)
    for i in metal_list[1:]:
        onedata = load_onemetal_data(i,solvent)
        final_data = pd.concat([final_data,onedata],axis=0)
    return final_data

my_net = torch.load('m_1.pkl')
my_net.eval()
all_metal_list = ['23V', '77Ir', '45Rh', '22Ti', '42Mo', '73Ta', '44Ru', '78Pt', '74W', '25Mn', '24Cr', '27Co', '75Re', '76Os', '26Fe']

solvents_list = ['acetonitrile', 'dmf', 'ethanol']
solvent = solvents_list[0]
print('solvent: ', solvent)

split_index = 12

transfer_data = merge_data(all_metal_list[split_index:],solvent)

transfer_x = transfer_data.iloc[:, :5000].values

transfer_all_y = transfer_data.iloc[:, 5000:5007].values  # CHO,CHOH,CO,COH,COOH,OCH3,OCHO

transfer_x = torch.tensor(transfer_x, dtype=torch.float32)
transfer_all_y = torch.tensor(transfer_all_y, dtype=torch.float32)

outputs = my_net(transfer_x)
accuracy_list = []
for idx in range(7):
    true = transfer_all_y[:, idx]
    out = outputs[:, idx]
    predict = []
    for i in out:
        if i > 0:
            predict.append(1)
        else:
            predict.append(0)
    result1 = confusion_matrix(true, predict)
    print(f'---------------{idx}---------------')  # 0: CHO 1: CHOH 2: CO 3: COH 4: COOH 5: OCH3 6: OCHO
    print("Confusion Matrix:")
    print(result1)
    result2 = accuracy_score(true, predict)
    print("Accuracy:", result2)
    accuracy_list.append(result2)
print('average_accuracy:', sum(accuracy_list) / len(accuracy_list))
print('all_metal_list', all_metal_list)
print('solvent: ', solvent)