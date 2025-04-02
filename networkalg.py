""" Node Classification in Graph Neural Networks

This algorithm doesn't use pytorch geometric to create the network and is handmade but it uses PyG for some utility functions. 
This algorithm looks at 8 nodes which each contain one feature with one classifying target which is either 0 or 1. There is also an
edge list which shows which nodes interact. Then the edges are converted from an edge list to an adjacency matrix. Then the model is 
created with two classes with one being the GCN model and one being the GCN layer which the parent of the GCN model. Next the model is run by 
creating neural network with three layers: one being the in put layer with 1 node, one being the hidden layer with 16 nodes, and one being the 
output layer with 2 nodes. The first layer is run which takes the features from each node and expands them into 16 different numbers each and the 
adjacency matrix is then used to create a weight towards the interactions shown. After the reLu is run which changes all negative values to 0 in the features
Finally the second layer is run which compresses the 16 features in each node to 2 features in each node. Then an error calculation is made to see how far off
the predicted value is to the expected value and then gradient descent is done to get the error closer to 0. In the end this is repeated 1000 times until the predicted values
is close to the expected value.
"""

import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from tqdm import tqdm

## Data ##
#dataset = Planetoid(root = ".", name = "Cora")
x = torch.tensor([[16],[14],[11],[15],[45],[40],[42],[48]], dtype = torch.float)
edge_index = torch.tensor([[0,0,0,1,1,2,4,4,4,5,5,6],
                           [1,2,3,2,3,3,5,6,7,6,7,7]], dtype = torch.long)
y = torch.tensor([0,0,0,0,1,1,1,1])
data = Data(x = x, edge_index = edge_index, y = y)
print(data)
dataAdj = to_dense_adj(data.edge_index)[0] # converting an edge list to an adjacency matrix
dataAdj = torch.eye(dataAdj.shape[0]) + dataAdj
dataAdj

G = to_networkx(data, to_undirected = True)
#pos = nx.spring_layout(G, seed=0)
nx.draw(G, with_labels = True)

## Model ##
class myGCNLayer(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.lin1 = nn.Linear(dim_in, dim_out) # mainly addition and mutliplication
  def forward(self, x, adj):
    x = self.lin1(x) # each node is expanded from 1 feature to 16 features per node. The number 16 comes from the variable dim_hidden
    x = adj @ x
    return x

class myGCN(nn.Module):
  def __init__(self, dim_in, dim_hidden, dim_out):
    super().__init__()
    self.layer1 = myGCNLayer(dim_in, dim_hidden)
    self.layer2 = myGCNLayer(dim_hidden, dim_out)
  def forward(self, x, adj):
    x = self.layer1(x, adj)
    x = torch.relu(x)
    x = self.layer2(x, adj)
    return x
model = myGCN(data.num_features, 16, 2)

## Training ##
lossList = []
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in tqdm(range(1000)):
    pred = model(data.x, dataAdj)
    #loss = lossfn(pred[data.train_mask], data.y[data.train_mask]) # Use this for the Cora and KarateClub data
    loss = lossfn(pred, data.y) # Use this for the PPI data
    loss.backward() # Computes the derivative/slope of the error
    optimizer.step() # Takes the step towards the local min using the lr
    optimizer.zero_grad() # Resets the slope calculation in order for change to occur
    lossList.append(loss.detach())
plt.plot(lossList)

## Testing ##
ypred = model(data.x, dataAdj).argmax(axis = 1)
print(ypred)
print(data.y)
print(accuracy_score(ypred, data.y))
