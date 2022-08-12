import torch
from train_teacher import Net
import numpy as np
import gzip
from torch_geometric.data import Data
from torch_geometric.data import Data, DataLoader


batch_size = 100
device = torch.device('cuda')
teacher = Net().to(device)
teacher.load_state_dict(torch.load(r'./teacher_params/epoch152.pth'))
teacher.eval()


def load_mnist_graph(data_size=10000):
    data_list = []
    labels = 0
    with gzip.open('./dataset/fashion-mnist/raw/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        #labels = labels[0:9000]
        # print(labels.shape)      # (10000,)

    for i in range(data_size):
        edge = torch.tensor(
            np.load('./dataset/edge.npy').T,   # 转置
            dtype=torch.long)
        x = torch.tensor(
            np.load('./dataset/fashion-mnist/images/test_data_coors/' + str(i) + '.npy'),
            dtype=torch.float)
        # print(x)

        d = Data(x=x, edge_index=edge.contiguous(), t=int(labels[i]))
        data_list.append(d)
        if i % 1000 == 999:
            print("\rData loaded " + str(i + 1), end="  ")

    print("Complete!")
    return data_list


test_mnist_list = load_mnist_graph()
test_set = test_mnist_list
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = teacher(data)
        _, predicted = torch.max(outputs, 1)
        total += data.t.size(0)
        correct += (predicted == data.t).sum().cpu().item()
test_acc = correct / total
print(total)
print('Test accuracy is %.2f%%' % float(test_acc * 100))
