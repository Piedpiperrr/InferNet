import torch
import numpy as np
from torch.utils.data import Dataset
from InferNet import get_net
from tqdm import tqdm
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def default_loader(array_path):
    temp = np.load(array_path)
    temp_tensor = torch.Tensor(temp)
    return temp_tensor


# test data path (H)   shape(784, 1)
test_file_path = []
for i in range(1000):
    features_path = r"./dataset/fashion-mnist/images/test_data/{}.npy".format(i)
    test_file_path.append(features_path)


# train data embedding (E)    shape(784, 128)
test_data_embedding = []
for i in range(1000):
    embedding_array = np.load(
        r"./dataset/fashion-mnist/images/test_data_embeddings/{}.npy".format(i))
    embedding_tensor = torch.Tensor(embedding_array)
    test_data_embedding.append(embedding_tensor)


class TestSet(Dataset):
    def __init__(self, loader=default_loader):
        self.embeddings = test_data_embedding
        self.target_path = test_file_path
        self.loader = loader

    def __getitem__(self, index):
        embed = self.embeddings[index]
        f_path = self.target_path[index]
        feats = self.loader(f_path)
        return embed, feats

    def __len__(self):
        return len(self.embeddings)


test_set = TestSet()
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
net = get_net()


state_dict = torch.load(r"./InferNet_autoencoder.pth", map_location=device)
net.load_state_dict(state_dict)
net.eval()


def MSE(embedding, label):
    err = np.sum((embedding - label)**2)
    return err


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true)


print('\n')
print('test:')
result_mse = 0
result_mae = 0
result_mape = 0
i = 0
for embeddings, features in tqdm(test_loader):
    embeddings = embeddings.to(device).float()
    features = features.to(device).float()
    features = features.view(784, 1)
    embeddings = embeddings.view(-1, 1, 784, 128)
    outputs = net(embeddings)
    outputs = outputs.view(784, 1)
    outputs = outputs.cpu().detach().numpy()
    np.save(r'./pred_autoencoder/{}_pred'.format(i), outputs)
    features = features.cpu().detach().numpy()
    mse = mean_squared_error(outputs, features)
    result_mse += mse
    mae = mean_absolute_error(outputs, features)
    result_mae += mae
    mape = mean_absolute_percentage_error(outputs, features)
    result_mape += mape
    i += 1
RMSE = math.sqrt(result_mse / len(test_set))
MAE = result_mae / len(test_set)
MAPE = result_mape / len(test_set) * 100
print('RMSE:', RMSE)
print('MAE:', MAE)
#print('MAPE:', MAPE)
