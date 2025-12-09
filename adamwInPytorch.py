import torch
import torch.nn as nn
from torch,optim import AdamW

# model example
model = nn.Linear(10, 1)

# create optimizator AdamW
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)

# example of learning step 
loss = torch.nn.functional.mse_loss(model(torch.rand(1, 10)), torch.randn(1, 1))
loss.backward()
optimizer.step()
optimizer.zero_grad()