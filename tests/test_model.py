import torch
import torch.nn.functional as F
from src.model import GCN

def test_gcn_output_shape():
    """Tests if the forward pass mathematical matrix dimensions are correct."""
    # Arranging Create a tiny dummy graph
    num_nodes = 10
    num_features = 16
    num_classes = 7
    
    # Dummy node features
    x = torch.randn((num_nodes, num_features))
    
    # Dummy edges (Adjacency list telling PyG how nodes connect)
    edge_index = torch.tensor([[0, 1, 2, 3], 
                               [1, 2, 3, 4]], dtype=torch.long)
                               
    # 2. Act: Initialize model and push the dummy graph through
    model = GCN(num_features=num_features, hidden_channels=8, num_classes=num_classes)
    out = model(x, edge_index)
    
    # 3. Assert: The output must be [10 nodes, 7 class probabilities]
    assert out.shape == (num_nodes, num_classes)

def test_gcn_backpropagation():
    """Tests if the calculus chain rule (gradients) reaches the model weights."""
    num_nodes = 5
    num_classes = 3
    
    x = torch.randn((num_nodes, 10))
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y = torch.randint(0, num_classes, (num_nodes,))
    
    model = GCN(num_features=10, hidden_channels=8, num_classes=num_classes)
    
    # Do a forward pass and calculate a fake loss
    model.train()
    out = model(x, edge_index)
    loss = F.nll_loss(out, y)
    
    # Trigger the calculus backward pass
    loss.backward()
    
    # Assert that at least one weight matrix has received a gradient update
    has_gradients = any(param.grad is not None for param in model.parameters())
    assert has_gradients == True