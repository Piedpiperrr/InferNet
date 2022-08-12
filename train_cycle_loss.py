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
    path = r'/home/XinyangLi/fashion_MNIST_GNN/dataset/generated_imgs/{}.npy'.format(i)
    train_data_path.append(path)


# train data embedding (E)    shape(784, 128)
train_data_embedding = []
for i in range(10000):
    embedding_array = np.load(r"/home/XinyangLi/fashion_MNIST_GNN/dataset/generated_embeddings/{}.npy".format(i))
    embedding_tensor = torch.Tensor(embedding_array)
    train_data_embedding.append(embedding_tensor)


class TrainSet(Dataset):
    def __init__(self, loader=default_loader):
        self.features_path = train_data_path
        self.embeddings = train_data_embedding
        self.loader = loader

    def __getitem__(self, index):
        f_path = self.features_path[index]
        features = self.loader(f_path)
        graph_embedding = self.embeddings[index]
        graph_embedding_copy = graph_embedding      # (784, 128)
        graph_embedding = graph_embedding.view(1, 784, 128)
        return graph_embedding, features, graph_embedding_copy

    def __len__(self):
        return len(self.embeddings)


train_set = TrainSet()
train_db, val_db = torch.utils.data.random_split(train_set, [9000, 1000])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    train_db, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_db, batch_size=batch_size, shuffle=True, num_workers=0)


def validate(validate_loader, model, mse):
    model.eval()
    validate_loss = []
    with torch.no_grad():
        for input_data, target, _ in validate_loader:
            input_data = input_data.to(device).float()
            target = target.to(device).float()
            output = model(input_data)
            output = output.view(-1, 784, 1)
            v_loss = mse(output, target)
            validate_loss.append(v_loss.item())
    return np.mean(validate_loss)


InferNet = get_net()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(
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
batch_node_coordinates = batch_node_coordinates.cuda()


cycle_loss_list = []
val_loss_list = []
min_loss = 100000


for epoch in range(200):
    cycle_loss = 0
    loss_1 = 0.0
    loss_2 = 0.0
    for emb, feats, em in tqdm(train_loader):
        emb = emb.to(device).float()
        feats = feats.to(device).float()
        em = em.to(device).float()
        optimizer.zero_grad()
        outputs = InferNet(emb)
        outputs = outputs.view(-1, 784, 1)
        loss1 = criterion(outputs, feats)   # 重构损失
        loss_1 += loss1.item()
        x = torch.cat([batch_node_coordinates, outputs], 2)
        data_list = []
        for i in range(x.shape[0]):
            edge = torch.tensor(
                np.load('./dataset/edge.npy').T,
                dtype=torch.long)
            feature = x[i]
            d = Data(x=feature, edge_index=edge.contiguous())
            data_list.append(d)
        data_loader = DataLoader(
            data_list,
            batch_size=batch_size,
            shuffle=False)
        em = em.view(-1, 128)
        for bat_data in data_loader:
            bat_data = bat_data.cuda()
            outputs_T, embeddings_T = teacher(bat_data, out_feature=True)
            loss2 = criterion(embeddings_T, em)     # cycle_loss
            loss_2 += loss2.item()
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            cycle_loss += loss.item()
    print('Epoch: %d loss1: %.6f loss2: %.6f total_loss: %.6f' %
              (epoch+1, loss_1 / len(train_loader), loss_2 / len(train_loader), cycle_loss / len(train_loader)))
    cycle_loss_list.append(cycle_loss / len(train_loader))
    val_loss = validate(val_loader, InferNet, criterion)
    if val_loss < min_loss:
        min_loss = val_loss
        print('save InferNet')
        torch.save(InferNet.state_dict(), 'InferNet_wo_ba3.pth')
    val_loss_list.append(val_loss)
print('finish training')


# 画loss曲线
epochs = range(len(cycle_loss_list))
plt.plot(epochs, cycle_loss_list, 'b-', label='Train_loss')
#plt.plot(epochs, val_loss_list, 'ro-', label='Val_loss')
plt.legend()
plt.title('Training loss vs epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('Training_loss_wo_ba3.png')
plt.close()
