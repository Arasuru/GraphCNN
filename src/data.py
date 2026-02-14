import os
from torch_geometric.datasets import Planetoid

def get_cora_dataset():
    root_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    dataset = Planetoid(root=root_dir, name='Cora')
    data = dataset[0] # Cora only has one graph, so take the first element

    return dataset, data

if __name__ == "__main__":
    dataset, data = get_cora_dataset()
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")