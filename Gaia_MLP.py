import torch
from GridDataset import TableDataset
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from torch.utils.data import Dataset, SequentialSampler, BatchSampler
from MLP import MLP


class TableDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


batch_size = 10000
n_epoch = 100

data = h5py.File("/mnt/home/apricewhelan/projects/gaia-scratch/data/gaiadr3-apogee-bprp-Xy.hdf5","r")

X = data['X'][:]
y = data['y'][:]
clean_index = np.where(np.invert(np.isnan(y[:,2])))[0]
X = X[clean_index]
y = y[clean_index]

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
x_scaler = QuantileTransformer()
y_scaler = QuantileTransformer()
x_scaler.fit(X)
y_scaler.fit(y)

x_scale = torch.from_numpy(x_scaler.transform(X)).cuda().float()
y_scale = torch.from_numpy(y_scaler.transform(y)).cuda().float()

dataset = TableDataset(x_scale, y_scale)

sampler = BatchSampler(SequentialSampler(dataset), batch_size, drop_last=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=sampler)

model = MLP(22,6,128,5)
model.train().cuda().float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = 1e9

for epoch in range(n_epoch):
    # Write training loop for MLP
    running_loss = 0.0
    for i, data in enumerate(dataloader,0):
        x_, y_ = data
        optimizer.zero_grad()
        output = model(x_)
        loss = torch.mean((output - y_)**2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20}')
            running_loss = 0.0

def predict(x):
    output = model(x).detach().cpu().numpy() 
    return y_scaler.inverse_transform(output)

pred_y = predict(x_scale)

