"""PyTorch Regression Model for value prediction tasks."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class RegressionModel(nn.Module):
    """Flexible regression model for predicting continuous values."""
    
    def __init__(self, input_dim: int = 256, hidden_dims: list = None, output_dim: int = 1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_regression_model(
    model: RegressionModel,
    train_data: torch.utils.data.DataLoader,
    val_data: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train a regression model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_data:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_data:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_losses.append(loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            history["val_loss"].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
    
    return history


def predict_regression(model: RegressionModel, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Make regression predictions."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        outputs = model(x)
    return outputs.cpu()
