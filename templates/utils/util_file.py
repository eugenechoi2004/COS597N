import os
from datetime import datetime

def generate_folder_paths(model_name):
    root_results = "./results"
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_version = f"{model_name}_{date_string}"
    
    folder_path = os.path.join(root_results, folder_version)
    checkpoint_path = os.path.join(folder_path, "checkpoints")
    graph_path = os.path.join(folder_path, "debug_graphs")
    output_path = os.path.join(folder_path, "outputs")

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    return folder_path, checkpoint_path, graph_path, output_path