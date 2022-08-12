import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from InferNet import get_net
from train_teacher import Net
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def default_loader(array_path):
    temp = np.load(array_path)
    temp_tensor = torch.Tensor(temp)
    return temp_tensor


# training data path (H)   shape(784, 1)
train_data_path = []
for i in range(10000):
    path = r'./dataset/generated_imgs/{}.npy'.format(i)
    train_data_path.append(path)


# train data embedding (E)    shape(784, 128)
train_data_embedding = []
for i in range(10000):
    embedding_array = np.load(
    r"./dataset/generated_embeddings/{}.npy".format(i))
    embedding_tensor = torch.Tensor(embedding_array)
    train_data_embedding.append(embedding_tensor)


class TrainSet1(Dataset):
    def __init__(self, loader=default_loader):
        self.features_path = train_data_path
        self.loader = loader
        self.embeddings = train_data_embedding

    def __getitem__(self, index):
        f_path = self.features_path[index]
        features = self.loader(f_path)
        embedding = self.embeddings[index]
        embedding_copy = embedding
        embedding = embedding.view(1, 784, 128)
        return embedding, embedding_copy, features

    def __len__(self):
        return len(self.features_path)


# test data embedding (E_t)     shape:(784, 128)
test_data_embedding = []
for it in range(5000):
    test_embedding_array = np.load(
        './dataset/fashion-mnist/images/test_data_embeddings/{}.npy'.format(it))
    test_embedding_tensor = torch.Tensor(test_embedding_array)
    test_data_embedding.append(test_embedding_tensor)


class TestSet(Dataset):
    def __init__(self):
        self.embeddings = test_data_embedding
        self.target = test_data_embedding

    def __getitem__(self, item):
        embed = self.embeddings[item]
        embed = embed.view(1, 784, 128)
        target = self.embeddings[item]
        return embed, target

    def __len__(self):
        return len(self.embeddings)


train_set1 = TrainSet1()
test_set = TestSet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100
train_loader1 = torch.utils.data.DataLoader(
    train_set1,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)


InferNet = get_net()
criterion = nn.MSELoss()
optimizer_train = optim.RMSprop(
    InferNet.parameters(),
    lr=0.001,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0)


teacher = Net().to(device)  # cuda
teacher.load_state_dict(torch.load(r'./teacher_params/epoch152.pth'))
teacher.eval()

node_coordinates = []
for m in range(28):
    for n in range(28):
        node_coordinates.append([m, n])
node_coordinates = np.array(node_coordinates)  # shape : (784, 2)
batch_node_coordinates = []
for q in range(batch_size):
    batch_node_coordinates.append(node_coordinates)
batch_node_coordinates = np.array(batch_node_coordinates)
batch_node_coordinates = torch.from_numpy(batch_node_coordinates)
batch_node_coordinates = batch_node_coordinates.to(device)


cycle_loss1_list = []
test_cycle_loss_list = []

for epoch in range(1, 201):
    if epoch % 2 == 1:
        cycle_loss1 = 0
        for input1, input1_copy, target1 in tqdm(train_loader1):
            input1 = input1.to(device).float()
            input1_copy = input1_copy.to(device).float()
            target1 = target1.to(device).float()
            optimizer_train.zero_grad()
            outputs1 = InferNet(input1)
            outputs1 = outputs1.view(-1, 784, 1)
            loss1 = criterion(outputs1, target1)
            x1 = torch.cat([batch_node_coordinates, outputs1], 2)
            data_list_1 = []
            for i in range(x1.shape[0]):
                edge = torch.tensor(np.load('./dataset/edge.npy').T, dtype=torch.long)
                feature1 = x1[i]
                d1 = Data(x=feature1, edge_index=edge.contiguous())
                data_list_1.append(d1)
            data_loader1 = DataLoader(data_list_1, batch_size=batch_size, shuffle=False)
            input1_copy = input1_copy.view(-1, 128)
            for bat_data in data_loader1:
                bat_data = bat_data.to(device)
                outputs_1, embedding_1 = teacher(bat_data, out_feature=True)
                loss2 = criterion(embedding_1, input1_copy)
                loss = 1 * loss1 + 0.001 * loss2
                loss.backward()
                optimizer_train.step()
                cycle_loss1 += loss.item()
        print('Epoch: %d loss1: %.6f' % (epoch, cycle_loss1 / len(train_loader1)))
        cycle_loss1_list.append(cycle_loss1 / len(train_loader1))
    if epoch % 2 == 0:
        test_cycle_loss = 0
        for input3, target3 in tqdm(test_loader):
            input3 = input3.to(device).float()
            target3 = target3.to(device).float()
            optimizer_train.zero_grad()
            output3 = InferNet(input3)
            output3 = output3.view(-1, 784, 1)
            x3 = torch.cat([batch_node_coordinates, output3], 2)
            data_list_3 = []
            for i in range(x3.shape[0]):
                edge = torch.tensor(np.load('./dataset/edge.npy').T, dtype=torch.long)
                feature3 = x3[i]
                d3 = Data(x=feature3, edge_index=edge.contiguous())
                data_list_3.append(d3)
            data_loader3 = DataLoader(data_list_3, batch_size=batch_size, shuffle=False)
            target3 = target3.view(-1, 128)
            for bat_data3 in data_loader3:
                bat_data3 = bat_data3.to(device)
                outputs_3, embedding_3 = teacher(bat_data3, out_feature=True)
                loss3 = 1 * criterion(embedding_3, target3)
                loss3.backward()
                optimizer_train.step()
                test_cycle_loss += loss3.item()
        print('Epoch: %d loss3: %.6f' % (epoch, test_cycle_loss / len(test_loader)))
        test_cycle_loss_list.append(test_cycle_loss / len(test_loader))
        if epoch == 200:
            print('save InferNet')
            torch.save(InferNet.state_dict(), 'InferNet_gamma_0.001.pth')
print('finish training!')


# 画loss曲线
epochs_1 = range(100)
plt.plot(epochs_1, cycle_loss1_list, 'b-', label='Loss1')
plt.plot(epochs_1, test_cycle_loss_list, 'g-', label='Test cycle loss')
plt.legend()
plt.title('Training loss vs epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('Training_gamma_0.001.png')
plt.close()
