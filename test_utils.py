import requests
import numpy as np

def test_gcn_api():
    # Create a dummy graph with 2 nodes (papers) that cite each other
    # Each node needs 1433 random features to match the Cora dataset
    dummy_x = np.random.rand(2, 1433).tolist()
    dummy_edge_index = [[0, 1],   # Node 0 points to Node 1
                    [1, 0]]   # Node 1 points to Node 0

    # Send the JSON payload to your local FastAPI server
    print("Sending request to API...")
    response = requests.post("http://localhost:8000/predict", json={
        "x": dummy_x,
        "edge_index": dummy_edge_index
    })

    # Print the GCN's classifications!
    print("Response:", response.json())

if __name__ == "__main__":
    test_gcn_api()