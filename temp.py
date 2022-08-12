# Get fake MNIST images embeddings
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from train_teacher import Net



node_coordinates = []
for m in range(28):
    for n in range(28):
        node_coordinates.append([m, n])
node_coordinates = np.array(node_coordinates)  # shape : (784, 2)
batch_node_coordinates = []
for q in range(10000):
    batch_node_coordinates.append(node_coordinates)
batch_node_coordinates = np.array(batch_node_coordinates)
batch_node_coordinates = torch.from_numpy(batch_node_coordinates)
batch_node_coordinates = batch_node_coordinates.cuda()   # shape : (10000, 784, 2)

fake_imgs = np.load(r'/home/XinyangLi/fashion_MNIST_GNN/dataset/fashion-mnist/images/test_data.npy')
fake_imgs = fake_imgs.reshape(10000, -1, 1)   # shape : (10000, 784, 1)
fake_imgs = torch.from_numpy(fake_imgs)
fake_imgs = fake_imgs.cuda()
x = torch.cat([batch_node_coordinates, fake_imgs], 2)   # shape : (10000, 784, 3)
data_list = []
for i in range(x.shape[0]):
  edge = torch.tensor(np.load('./dataset/edge.npy').T, dtype=torch.long)
  feature = torch.tensor(x[i], dtype=torch.float)
  d = Data(x=feature, edge_index=edge.contiguous())
  data_list.append(d)
data_loader = DataLoader(data_list, batch_size=1, shuffle=False)
teacher = Net().cuda()  # cuda
teacher.load_state_dict(torch.load(r'./teacher_params/epoch152.pth'))
teacher.eval()
for i, batch_data in enumerate(data_loader):
  batch_data = batch_data.cuda()
  _, embedding_T = teacher(batch_data, out_feature=True)
  embedding_T = embedding_T.cpu().detach().numpy() # shape : (784, 128)
  np.save(r'/home/XinyangLi/fashion_MNIST_GNN/dataset/fashion-mnist/images/test_data_embeddings/{}'.format(i), embedding_T)