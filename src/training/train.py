import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def load_train_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Reads and parses the YAML training configuration."""
    # Step 1: Open and return parsed yaml dictionary
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config 
    

def train_model(config: dict, device: str = "auto") -> dict:
    """
    Initializes YOLO and runs fine-tuning using config dictionary.
    
    Returns:
        dict: Summary of training results and metrics.
    """
    # Step 2: Load model from config["model"]
    # Step 3: Run model.train(...) passing the relevant hyperparameters
    # Step 4: Run model.val() on the best checkpoint
    
    model = YOLO(config["model"])
    
    train_args = {k: v for k, v in config.items() if k != "model"}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.train(**train_args,device=device)

    
    val_results = model.val()

    metrics = {
        "precision": float(val_results.box.mp),      # Mean precision
        "recall": float(val_results.box.mr),         # Mean recall
        "map50": float(val_results.box.map50),       # mAP @ IoU 0.50
        "map50_95": float(val_results.box.map),      # mAP @ IoU 0.50:0.95
    }


    return metrics


if __name__ == "__main__":
    cfg = load_train_config("configs/train_config.yaml")
    print(f"Starting training for {cfg.get('epochs', 3)} epochs on model {cfg.get('model')}...")
    results = train_model(cfg)
    print("\nTraining complete! Validation metrics:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")