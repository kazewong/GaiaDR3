import torch
import numpy as np

model = torch.jit.load("serialized_MLP.pt")
scaler = np.load('./scaler.pickle', allow_pickle=True)

def predict(x):
    x = torch.from_numpy(scaler[0].transform(x)).float()
    output = model(x).detach().cpu().numpy() 
    return scaler[1].inverse_transform(output)

example = predict(np.random.uniform(size=(1,23)))