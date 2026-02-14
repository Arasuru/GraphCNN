import torch
import torch.nn.functional as F
from data import get_cora_dataset
from model import GCN

def train():
    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    print(f"Using device: {device}")
    dataset, data = get_cora_dataset()
    data = data.to(device)

    #Initializing Modelm and optimizer
    model = GCN(
        num_features = dataset.num_node_features,
        hidden_channels = 16,
        num_classes = dataset.num_classes 
    ).to_device()

    #Adam optimizer(adjusts learning rate for each weight individually)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    #Training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() #Reset gradients

        #Forward pass
        out = model(data.x, data.edge_index)

        #calculate loss only on training nodes(masking)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        #Backward pass and optimization step
        loss.backward()
        optimizer.step()

        #Evaluation every 10 epochs
        if epoch % 10 ==0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1) #Get predicted class labels

                correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                acc = int(correct) / int(data.val_mask.sum())
            
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Val Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()