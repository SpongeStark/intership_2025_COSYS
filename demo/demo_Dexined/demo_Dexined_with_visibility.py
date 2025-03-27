# %% [markdown]
# # Load packages

# %%
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import json
import numpy as np

# %%
import sys

sys.path.append("../../resource/DexiNed")
sys.path.append("../../src")

from model import DexiNed
from dataset import BIPEDv2

# %% [markdown]
# # Train

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 30
batch_size = 8
learning_rate = 1e-4
biped_dataset = BIPEDv2(
    "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/",
    "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real/"
)

model = DexiNed()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loader = DataLoader(biped_dataset, batch_size=4)
logging = {
    'train_loss': []
}


print(device)

# %%
model.train()
model = model.to(device)
for e in range(epoch):
    epoche_loss = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)[-1].squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print("*", end="", flush=True)
        epoche_loss.append(loss.detach().item())
    
    logging['train_loss'].append(np.mean(epoche_loss))
    print(f"\nIn epoch {e}, the average loss is {logging['train_loss'][-1]}")

# save files
model = model.to('cpu')
torch.save(model.state_dict(), "./checkpoints/cpt_visibility.pth")
print(f"Succeed saving mdoel parameters in ./checkpoints/cpt_visibility.pth.")
with open("./checkpoints/cpt_visibility.json", "w") as f:
    json.dump(logging, f)
print(f"Succeed saving log in ./checkpoints/cpt_visibility.json")

# %%
