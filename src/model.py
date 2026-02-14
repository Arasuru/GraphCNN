import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.module):

    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()

        #Layer 1: Input features to hidden representations
        self.conv1 = GCNConv(num_features, hidden_channels)

        #Layer 2: Hidden representations to output classes
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):

        #Layer 1: Apply GCNConv and ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        #Dropout layer for regularization to prevent overfitting
        x = F.dropout(x, training=self.training)

        #Layer 2: Apply GCNConv to get class scores
        x = self.conv2(x, edge_index)

        #log softmax to get log probabilities for each class
        return F.log_softmax(x, dim=1)
    