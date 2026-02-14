import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

#Defining the data model for incoming graph data using Pydantic
class GraphData(BaseModel):
    x: List[List[float]]  # Node features as a list of lists(list of nodes, each node with list of features)
    edge_index: List[List[int]]  # Edge indices as a list of lists(source nodes, target nodes)

#Initialize FastAPI app
app = FastAPI(title="GCN Cora Servicing API")

#global variable to hold the loaded model
global model

#choosing the latest model dynamically
@app.on_event("startup")
def load_latest_model():

    global model
    mlflow.set_tracking_uri("http://localhost:5000")  # Assuming MLflow tracking server is running locally

    experiment = mlflow.get_experiment_by_name("GCN_Cora_Experiment")
    if experiment is None:
        raise RuntimeError("No MLflow experiment found. Please run the training script first.")
    
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    if df.empty:
        raise RuntimeError("No runs found in the experiment. Please run the training script first.")

    run_id = df.iloc[0].run_id
    model_uri = f"runs:/{run_id}/gcn_cora_model"

    print(f"Loading model from MLflow run ID: {run_id}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval() # Set to evaluation mode (turns off dropout)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "architecture": str(model)
    }

# 4. The Prediction Endpoint
@app.post("/predict")
def predict(data: GraphData):
    try:
        # Step A: Convert the incoming JSON lists into PyTorch Math Tensors
        x_tensor = torch.tensor(data.x, dtype=torch.float)
        edge_index_tensor = torch.tensor(data.edge_index, dtype=torch.long)
        
        # Step B: Push the data through the GraphCNN
        with torch.no_grad():
            out = model(x_tensor, edge_index_tensor)
            
            # Convert log-probabilities to actual class predictions (0 through 6)
            predictions = out.argmax(dim=1).tolist()
            
        # Step C: Return the predictions to the user as JSON
        return {"predicted_classes": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))