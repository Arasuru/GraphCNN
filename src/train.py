import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from data import get_cora_dataset
from model import GCN

def train():
    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    print(f"Using device: {device}")
    dataset, data = get_cora_dataset()
    data = data.to(device)

    #parameters for tracking with mlflow
    params = {
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "hidden_channels": 16,
        "epochs": 200
    }
    #setting up local mlflow experiment
    mlflow.set_experiment("GCN_Cora_Experiment")

    #starting mlflow run to track parameters, metrics, and model artifacts
    with mlflow.start_run():
        mlflow.log_params(params)

        #Initializing Modelm and optimizer
        model = GCN(
            num_features = dataset.num_node_features,
            hidden_channels = 16,
            num_classes = dataset.num_classes 
        ).to(device)

        #Adam optimizer(adjusts learning rate for each weight individually)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params["learning_rate"], 
            weight_decay=params["weight_decay"])


        #Training loop
        epochs = params["epochs"]
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad() #Reset gradients

            #Forward pass
            out = model(data.x, data.edge_index)

            #calculate loss only on training nodes(masking)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

            #Backward pass and optimization step(Negative log likelihood loss is used for multi-class classification)
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

                #push metrics to mlflow server
                mlflow.log_metrics({
                    "train_loss": loss.item(),
                    "val_accuracy": acc
                }, step = epoch)

        #Save the trained model as an artifact in mlflow
        mlflow.pytorch.log_model(model, "gcn_cora_model")


if __name__ == "__main__":
    train()