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
    def __init__(self, input_size, hidden_size, latent_size):
        '''
        Initializes the layers of the graph neural network

        :param input_size: defines the original dimension of each node's feature vector
        :param hidden_size: defines size of the hidden dimension in the encoder and decoder
        :param latent_size: defines the dimensions of the latent space of the autoencoder
        '''

        super().__init__()
    
        self.encoder_layer_1 = GCNConv(input_size, hidden_size) #reduce features
        self.encoder_activation_1 = nn.ReLU()

        self.encoder_layer_2 = GCNConv(-1, latent_size) #reduce again to latent
        self.encoder_activation_2 = nn.ReLU()

        self.encoder_layer_3 = torch.nn.Linear(latent_size, latent_size)
        self.encoder_activation_3 = nn.Sigmoid()


        self.mu_layer = torch.nn.Linear(latent_size, latent_size) 
        self.logvar_layer = torch.nn.Linear(latent_size, latent_size)

        self.attr_decoder_layer_1 = GCNConv(-1, hidden_size)
        self.decoder_activation_1 = nn.ReLU()

        self.attr_decoder_layer_2 = GCNConv(-1, input_size)

        self.attr_decoder_layer_3 = torch.nn.Sigmoid()

        self.structure_decoder = DotProductDecoder(in_dim = input_size)

 
    
    def forward(self, x, edge_index):
        '''
        Performs the forward pass of the model

        :param x: the data being passed into the model, should be a tensor of size (number of nodes) x (feature vector length)
        :param edge_index: a tensor listing all of the edges as node pairings 

        :return s: the reconstructed adjacency matrix
        :return decoded: the reconstructed data
        :return mu: the mu parameter for the VAE's distribution
        :return logvar: the log of the variance parameter for the VAE's distribution
        '''
        x = self.encoder_layer_1(x, edge_index)
        x = self.encoder_activation_1(x)
        x = self.encoder_layer_2(x, edge_index)
        x = self.encoder_activation_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_activation_3(x)

        mu = self.mu_layer(x)

        logvar = self.logvar_layer(x)

        sigma = logvar.mul(0.5).exp()

        eps = torch.randn(mu.size())

        z = mu + sigma * eps

        s = self.structure_decoder(z)

        decoded = self.attr_decoder_layer_1(z, edge_index)
        decoded = self.decoder_activation_1(decoded)
        decoded = self.attr_decoder_layer_2(decoded, edge_index)

        return s, decoded, mu, logvar

class DotProductDecoder(nn.Module):
    def __init__(self, in_dim):
        '''
        Constructor for the inner product decoder

        :param in_dim: specifies the input dimension
        '''
        super(DotProductDecoder, self).__init__()
        self.in_dim = in_dim

    def forward(self, h):
        '''
        Forward pass for the inner product decoder
        Computes the dot product of the encoded input and its transpose to recreate the adjacency matrix 
        Applies a sigmoid activation function to every value in the adjacency matrix

        :param h: the encoded data

        :return edge_scores: the reconstructed adjacency matrix
        '''
        dot_product = torch.matmul(h, h.t())
        edge_scores = torch.sigmoid(dot_product)
        return edge_scores

def process_data(dataset_name):
    '''
    Function to preprocess data

    :param dataset_name: the name of the dataset to be loaded

    :return x: a tensor of size (number of nodes) x (feature vector length) made up of each node's feature vector
    :return edge_index: a tensor representing each of the edges in the graph as node pairings
    :return adj_mat: the graph's adjacency matrix
    :return y: a binary tensor of size (number of nodes) representing whether each node is an outlier or not
    '''
    data = load_data(dataset_name)
    x, edge_index, y = data.x, data.edge_index, data.y.bool()
    adj_mat = create_adj_mat(edge_index)
    return x, edge_index, adj_mat, y

def create_adj_mat(edge_index):
    '''
    Creates the graph's adjacency matrix from the edge index

    :param edge_index: a tensor representing each of the edges in the graph as node pairings

    :return adjacency_matrix: the graph's adjacency matrix
    '''
    num_nodes = edge_index.max().item() + 1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    return adjacency_matrix

x, edge_index, adj_mat, y = process_data('Enron')

def loss_fun(x, decoded, adj_mat, s, mu, logvar, alpha = 0.5, beta = 0.5,train = True):
    '''
    The loss function calculates the loss to be backpropogated in the model.  
    It combines MSE for the features and adjacency matrix along with KL divergence

    :param x: a tensor of size (number of nodes) x (feature vector length) made up of each node's feature vector
    :param decoded: the VAE's reconstruction of x
    :param adj_mat: the graph's adjacency matrix
    :param s: the VAE's reconstruction of the adjacency matrix
    :param mu: the mean for the distribution created in the latent space
    :param logvar: the log of the variance for the distribution in the latent space
    :param alpha: a weight to control the percentage of structure loss (and feature loss)
    :param beta: a weight to control the percentage of feature loss (and KL divergence)
    :param train: a boolean to communicate if the model is training or testing

    :return loss: a loss value for each node in the form of a tensor with shape (number of nodes)
    '''
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    feature_loss = torch.pow(x - decoded, 2) #MSE from oroginal and reconstructed features
    structure_loss = torch.pow(adj_mat - s, 2) #MSE from original adj matrix and reconstructed adj mat
    attribute_loss = beta * feature_loss + (1 - beta) * kl_loss #aggregated loss for node attributes 
    structure_loss = torch.sqrt(torch.sum(structure_loss, 1)) #calculate adj matrix MSE by node

    if train == False:
        #during testing, don't include KL divergence
        feature_loss = torch.mean(feature_loss, dim=1, keepdim=True)
        loss = alpha * structure_loss + (1 - alpha) * feature_loss.squeeze()
    else:
        attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
        loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()
    return loss
    

def train(model, x, edge_index, optimizer, adj_mat, n_epochs = 100):
    '''
    The train function executes the forward passes and backpropogation that train the model

    :param model: the model to be trained
    :param x: the tensor of the node features
    :param edge_index: a tensor representing each of the edges in the graph as node pairings
    :param optimizer: the function that determines how weights are updated in the network
    :param adj_mat: the graph's adjacency matrix
    :param n_epoch: specifies the number of epochs for the model to be trained on

    :return model: the model with updated weights
    '''
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
    '''
    The test function performs a single forward pass, then computes the auc score based on the loss value for each node

    :param model: the model to be trained
    :param x: the tensor of the node features
    :param edge_index: a tensor representing each of the edges in the graph as node pairings
    :param adj_mat: the graph's adjacency matrix

    :return: auc score which is an accuracy score for binary classification
    '''
    model.eval()
    s, decoded, mu, logvar = model(x, edge_index)
    loss = loss_fun(x, decoded, adj_mat, s, mu, logvar, train=False)
    loss_per_node = double_recon_loss(x, decoded, adj_mat, s, weight=0.5)
    auc_score = eval_roc_auc(y, loss.detach())
    print("My Score:")
    return auc_score
    

model = GVAE(input_size=x.size()[1], hidden_size = (x.size()[1] // 2), latent_size=(x.size()[1] // 4))
optimizer = torch.optim.Adam(model.parameters(), lr = .01)

model = train(model, x, edge_index, optimizer, adj_mat)
print(test(model, x, edge_index, adj_mat))




