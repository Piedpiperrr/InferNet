import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from train_teacher import Net
import gzip
import torch
from torch.autograd import Variable
from load_mnist_graph import load_mnist_graph
import torch.nn as nn


img_size = 28
latent_dim = 100
channels = 1
batch_size = 512
n_epochs = 200
lr_G =1e-6
lr_S = 4e-3
oh = 0.5
ie = 0.1
a = 0.01


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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(
                latent_dim,
                128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


generator = Generator().cuda()  # cuda
teacher = Net().cuda()  # cuda
teacher.load_state_dict(torch.load(r'./teacher_params/epoch152.pth'))
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()  # cuda


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    #l_kl = F.kl_div(p, q, size_average=False) / y.shape[0]
    l_kl = F.kl_div(p, q, reduction='sum') / y.shape[0]

    return l_kl


student = Net().cuda()
data_test_list = load_mnist_graph()
data_test_loader = DataLoader(data_test_list, batch_size=100, shuffle=True)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)
optimizer_S = torch.optim.Adam(student.parameters(), lr=lr_S)


for epoch in range(n_epochs):
    total_correct = 0
    avg_loss = 0.0
    for j in range(120):
        student.train()
        z = Variable(torch.randn(batch_size, latent_dim)).cuda()  # cuda
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()
        gen_imgs = generator(z)  # (512, 1, 28, 28)
        gen_imgs = gen_imgs.view(batch_size, -1, 1)  # (512, 784, 1)
        x = torch.cat([batch_node_coordinates, gen_imgs], 2)  # shape : (512, 784, 3)
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
            shuffle=True)
        for batch in data_loader:
            batch = batch.cuda()
            outputs_T, features_T = teacher(batch, out_feature=True)
            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T, pred)
            softmax_o_T = torch.nn.functional.softmax(
                outputs_T, dim=1).mean(dim=0)
            loss_information_entropy = (
                softmax_o_T * torch.log10(softmax_o_T)).sum()
            loss = loss_one_hot * oh + loss_information_entropy * \
                ie + loss_activation * a
            loss_kd = kdloss(student(batch.detach()), outputs_T.detach())
            loss += loss_kd
            loss.backward()
            optimizer_G.step()
            optimizer_S.step()

            if j == 1:
                print(
                    "[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" %
                    (epoch + 1,
                     n_epochs,
                     loss_one_hot.item(),
                     loss_information_entropy.item(),
                     loss_activation.item(),
                     loss_kd)
                )
    
    with torch.no_grad():
        for batch_data in data_test_loader:
            batch_data = batch_data.cuda()
            student.eval()
            out = student(batch_data)
            avg_loss += criterion(out, batch_data.t)
            _, predicted = torch.max(out, 1)
            total_correct += (predicted == batch_data.t).sum().cpu().item()

    avg_loss /= len(data_test_list)
    print('Test Avg. Loss: %f, Accuracy: %f' %
          (avg_loss.data.item(), float(total_correct) / len(data_test_list)))
    torch.save(generator.state_dict(),'./generator_params/epoch{}.pth'.format(epoch + 1))
