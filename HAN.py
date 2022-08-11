import os.path as osp
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
import os

import torch_geometric.transforms as T
from torch_geometric.nn import HANConv

from torchmetrics import ConfusionMatrix

if os.path.exists('MIMICDataObj.pt'):
    data =torch.load("MIMICDataObj.pt")
print(data)


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['Admission'])
        return out


model = HAN(in_channels=-1, out_channels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['Admission'].train_mask
    loss = F.cross_entropy(out[mask], data['Admission'].y[mask],weight=torch.tensor([0.22, 0.78]).to(device))  #,weight=torch.tensor([0.20, 0.80]).to(device)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(epoch) -> List[float]:
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['Admission'][split]
        acc = (pred[mask] == data['Admission'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
        if epoch%50==0:
            confmat = ConfusionMatrix(num_classes=2)
            print(split,confmat(pred[mask].cpu(), data['Admission'].y[mask].cpu()))
    return accs


best_val_acc = 0
start_patience = patience = 100
for epoch in range(1, 201):

    loss = train()
    train_acc, val_acc, test_acc = test(epoch)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    if best_val_acc <= val_acc:
        patience = start_patience
        best_val_acc = val_acc
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break
