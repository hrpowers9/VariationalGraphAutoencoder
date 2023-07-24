import pygod
from pygod.utils import load_data
from pygod.detector import DOMINANT
from pygod.detector import AdONE
import torch
from pygod.metric import eval_roc_auc
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils.convert
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCN
import torch_sparse
from pygod.nn.functional import double_recon_loss
import torch.nn as nn
import numpy as np

class GVAE(torch.nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        latent_size = 14
        print(input_size // 2)
        print(latent_size)
        self.encoder_layer_1 = GCNConv(input_size, 16) #reduce features
        self.encoder_activation_1 = nn.ReLU()

        self.encoder_layer_2 = GCNConv(-1, latent_size) #reduce again to latent
        self.encoder_activation_2 = nn.ReLU()

        self.encoder_layer_3 = torch.nn.Linear(latent_size, latent_size)
        self.encoder_activation_3 = nn.Sigmoid()


        self.mu_layer = torch.nn.Linear(latent_size, latent_size) 
        self.logvar_layer = torch.nn.Linear(latent_size, latent_size)

        self.attr_decoder_layer_1 = GCNConv(-1, 16)
        self.decoder_activation_1 = nn.ReLU()

        self.attr_decoder_layer_2 = GCNConv(-1, input_size)

        self.attr_decoder_layer_3 = torch.nn.Sigmoid()

        self.structure_decoder = DotProductDecoder(in_dim = input_size)

 
    
    def forward(self, x, edge_index):
        x = self.encoder_layer_1(x, edge_index)
        x = self.encoder_activation_1(x)
        x = self.encoder_layer_2(x, edge_index)
        x = self.encoder_activation_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_activation_3(x)

        mu = self.mu_layer(x)
        # print(mu.size()) #number of nodes x latent 
        # print("mean")
        # print(mu)
        # print(mu.mean())
        # print("mean")

        logvar = self.logvar_layer(x)

        sigma = logvar.mul(0.5).exp()
        # print("variance")
        # print(sigma)
        # print(sigma.mean())
        # print("variance")

        eps = torch.randn(mu.size())

        z = mu + sigma * eps
        # print("sample")
        # print(z)

        s = self.structure_decoder(z)

        decoded = self.attr_decoder_layer_1(z, edge_index)
        decoded = self.decoder_activation_1(decoded)
        decoded = self.attr_decoder_layer_2(decoded, edge_index)
        # decoded = self.attr_decoder_layer_3(decoded)

        return s, decoded, mu, logvar

class DotProductDecoder(nn.Module):
    def __init__(self, in_dim):
        super(DotProductDecoder, self).__init__()
        self.in_dim = in_dim

    def forward(self, h):
        dot_product = torch.matmul(h, h.t())
        edge_scores = torch.sigmoid(dot_product)
        return edge_scores

def process_data(dataset_name):
    data = load_data(dataset_name)
    x, edge_index, y = data.x, data.edge_index, data.y.bool()
    adj_mat = create_adj_mat(edge_index)
    return x, edge_index, adj_mat, y, data

def create_adj_mat(edge_index):
    num_nodes = edge_index.max().item() + 1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    return adjacency_matrix

x, edge_index, adj_mat, y, data = process_data('Enron')

def loss_fun(x, decoded, adj_mat, s, mu, logvar, alpha = 0.5, train = True):
    structure_loss = torch.pow(adj_mat - s, 2) #MSE from original adj matrix and reconstructed adj mat
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    feature_loss = torch.pow(x - decoded, 2) #MSE from oroginal and reconstructed features
    attribute_loss = feature_loss + kl_loss 
 
    if train == False:
        structure_loss = torch.sqrt(torch.sum(structure_loss, 1))
        feature_loss = torch.mean(feature_loss, dim=1, keepdim=True)
        loss = alpha * structure_loss + (1 - alpha) * feature_loss.squeeze()
    else:
        attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
        structure_loss = torch.sqrt(torch.sum(structure_loss, 1))
        loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()
    return loss
    

def train(model, x, edge_index, optimizer, adj_mat, n_epochs = 100):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        s, decoded, mu, logvar = model(x, edge_index) #forward pass   
        loss = loss_fun(x, decoded, adj_mat, s, mu, logvar, train=True)
        loss = loss.mean()
        loss.backward() # backpropogation
        optimizer.step()
        print(f"Epoch [{epoch}/{n_epochs}], Train Loss: {loss.item():.4f}")
    return model


def test(model, x, edge_index, adj_mat):
    model.eval()
    s, decoded, mu, logvar = model(x, edge_index)
    loss = loss_fun(x, decoded, adj_mat, s, mu, logvar, train=False)
    loss_per_node = double_recon_loss(x, decoded, adj_mat, s, weight=0.5)
    auc_score = eval_roc_auc(y, loss.detach())
    print("My Score:")
    return auc_score
    

model = GVAE(input_size=x.size()[1], latent_size=(x.size()[1] // 4))
optimizer = torch.optim.Adam(model.parameters(), lr = .01)

print(data)
model = train(model, x, edge_index, optimizer, adj_mat)
print(test(model, x, edge_index, adj_mat))

# given_dominant = DOMINANT(epoch=100)
# given_dominant.fit(data)

# pred, score, prob, conf = given_dominant.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
# print('Labels:')
# print(pred)

# print('Raw scores:')
# print(score)

# print('Probability:')
# print(prob)

# print('Confidence:')
# print(conf)

# auc_score = eval_roc_auc(data.y, score)
# print('Benchmark Score:', auc_score)



