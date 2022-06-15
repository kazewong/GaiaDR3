import torch
from GridDataset import TableDataset
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from scipy.stats import binned_statistic_2d
from torch.utils.data import Dataset, SequentialSampler, BatchSampler
from MLP import MLP
import copy
import matplotlib.pyplot as plt

class TableDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


batch_size = 1000
n_epoch = 1000

data = h5py.File("/mnt/home/apricewhelan/projects/gaia-scratch/data/gaiadr3-apogee-bprp-Xy.hdf5","r")

X_ = data['X'][:]
Y_ = data['y'][:]
clean_index = np.where(np.invert(np.isnan(Y_[:,2])))[0]
X_ = X_[clean_index]
Y_ = Y_[clean_index]

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
x_scaler = QuantileTransformer()
Y_scaler = QuantileTransformer()
x_scaler.fit(X_)
Y_scaler.fit(Y_)

x_scale = torch.from_numpy(x_scaler.transform(X_)).cuda().float()
y_scale = torch.from_numpy(Y_scaler.transform(Y_)).cuda().float()

dataset = TableDataset(x_scale, y_scale)

length = torch.tensor([int(dataset.__len__()*0.8),dataset.__len__()-int(dataset.__len__()*0.8)])
train_data, val_data = torch.utils.data.random_split(dataset, length)

sampler = BatchSampler(SequentialSampler(train_data), batch_size, drop_last=False)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=None, sampler=sampler)

sampler = BatchSampler(SequentialSampler(val_data), batch_size, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=None, sampler=sampler)

model = MLP(22,6,128,5)
model.train().cuda().float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_train_loss = 1e9
best_valid_loss = 1e9
best_model = copy.copy(model)
best_valid_model = copy.copy(model)

for epoch in range(n_epoch):
    # Write training loop for MLP
    running_loss = 0.0
    val_loss = 0.0
    for i, data in enumerate(train_dataloader,0):
        x_, y_ = data
        optimizer.zero_grad()
        output = model(x_)
        loss = torch.mean((output - y_)**2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 20 == 19:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20}')
        #     running_loss = 0.0

    print(f'Epoch {epoch + 1} training loss: {running_loss / i}')

    # Write validation loop for MLP
    for i, data in enumerate(val_dataloader,0):
        x_, y_ = data
        output = model(x_)
        loss = torch.mean((output - y_)**2)
        val_loss += loss.item()
    print(f'[{epoch + 1}] val_loss: {val_loss / len(val_dataloader)}')

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_model = copy.copy(model)


def predict(x):
    output = best_valid_model(x).detach().cpu().numpy() 
    return Y_scaler.inverse_transform(output)

pred_y = predict(x_scale)
error = pred_y - Y_

def plot_corner(X,y,label_idx = 0):
    feature_idx = np.arange(0, 5)


    maxn = len(feature_idx)
    vmin, vmax = np.nanpercentile(y[:, label_idx], [1, 99])

    fig, axes = plt.subplots(
        maxn, maxn, 
        figsize=(13, 12), 
        constrained_layout=True,
    #     sharex='col', sharey='row'
    )

    for axi, i in enumerate(feature_idx):
        for axj, j in enumerate(feature_idx):
            ax = axes[axj, axi]

            if i > j:
                ax.set_visible(False)
                continue

            if i == j:
                xx, yy = X[:, i], y[:, label_idx]
                _mask = np.isfinite(xx) & np.isfinite(yy)
                H, xe, ye = np.histogram2d(
                    xx[_mask], 
                    yy[_mask],
                    bins=(
                        np.linspace(*np.nanpercentile(xx[_mask], [0.1, 99.9]), 128),
                        np.linspace(*np.nanpercentile(yy[_mask], [0.1, 99.9]), 128)
                    ),
                )

                ax.pcolormesh(xe, ye, H.T, cmap='Greys', norm=mpl.colors.LogNorm())

            else:
                xx, yy = X[:, i], X[:, j]
                zz = y[:, label_idx]
                _mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz)

                stat = binned_statistic_2d(
                    xx[_mask], 
                    yy[_mask],
                    zz[_mask],
                    bins=(
                        np.linspace(*np.nanpercentile(xx[_mask], [0.1, 99.9]), 128),
                        np.linspace(*np.nanpercentile(yy[_mask], [0.1, 99.9]), 128)
                    ),
                    statistic=np.nanmean
                )

                cs = ax.pcolormesh(stat.x_edge, stat.y_edge, stat.statistic.T, 
                                cmap='magma_r', 
                                vmin=vmin, vmax=vmax)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    cb = fig.colorbar(cs, ax=axes, aspect=30)
    fig.savefig('./plots/corner'+str(label_idx)+'.png')

for i in range(6):
    plot_corner(X_,pred_y,i)