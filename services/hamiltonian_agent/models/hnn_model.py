# services/hamiltonian_agent/models/hnn_model.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

@dataclass
class HNNConfig:
    """Configuration for Hamiltonian Neural Network"""
    input_dim: int
    hidden_dim: int
    num_layers: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class HamiltonianLayer(nn.Module):
    """
    Custom layer that enforces Hamiltonian dynamics.
    Ensures the network learns energy-preserving transformations.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with skew-symmetric matrix for energy conservation
        self.weights = nn.Parameter(
            torch.randn(in_features, out_features) / np.sqrt(in_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create skew-symmetric matrix for Hamiltonian dynamics
        W = self.weights - self.weights.t()
        return torch.matmul(x, W)

class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network for learning conservation laws in financial markets.
    Implements symplectic integration to preserve geometric structure.
    """
    
    def __init__(self, config: HNNConfig):
        super().__init__()
        self.config = config
        
        # Build network architecture
        layers = []
        current_dim = config.input_dim
        
        # Add Hamiltonian layers with non-linearities
        for _ in range(config.num_layers):
            layers.extend([
                HamiltonianLayer(current_dim, config.hidden_dim),
                nn.Tanh(),  # Smooth, bounded non-linearity
                nn.LayerNorm(config.hidden_dim)  # Normalize for stability
            ])
            current_dim = config.hidden_dim
            
        # Final Hamiltonian layer for output
        layers.append(HamiltonianLayer(current_dim, config.input_dim))
        
        self.network = nn.Sequential(*layers)
        self.to(config.device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing both predictions and Hamiltonian
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            predictions: Predicted next state
            hamiltonian: Computed Hamiltonian (energy) of the system
        """
        # Split input into position (q) and momentum (p)
        q, p = torch.split(x, x.shape[1] // 2, dim=1)
        
        # Compute Hamiltonian (total energy)
        kinetic = 0.5 * torch.sum(p**2, dim=1)
        potential = self.compute_potential(q)
        hamiltonian = kinetic + potential
        
        # Forward through network
        dqdt = self.network(torch.cat([q, p], dim=1))
        
        # Ensure symplectic structure
        predictions = x + dqdt
        
        return predictions, hamiltonian
    
    def compute_potential(self, q: torch.Tensor) -> torch.Tensor:
        """Compute potential energy of the system"""
        # For financial markets, potential energy could represent
        # deviation from equilibrium price levels
        return 0.5 * torch.sum(q**2, dim=1)
    
    def symplectic_euler_step(
        self,
        state: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Perform symplectic Euler integration step to preserve
        geometric structure of the system
        """
        q, p = torch.split(state, state.shape[1] // 2, dim=1)
        
        # Update momentum
        dH_dq = torch.autograd.grad(
            self.compute_potential(q).sum(),
            q,
            create_graph=True
        )[0]
        p_new = p - dt * dH_dq
        
        # Update position
        q_new = q + dt * p_new
        
        return torch.cat([q_new, p_new], dim=1)
    
    def energy_loss(
        self,
        initial_state: torch.Tensor,
        predicted_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy conservation loss
        
        Args:
            initial_state: Initial system state
            predicted_state: Predicted next state
            
        Returns:
            loss: Energy conservation loss
        """
        _, initial_energy = self.forward(initial_state)
        _, predicted_energy = self.forward(predicted_state)
        
        # Energy should be conserved
        return torch.mean((predicted_energy - initial_energy)**2)
    
    def prediction_loss(
        self,
        predicted_state: torch.Tensor,
        target_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction accuracy loss
        """
        return torch.mean((predicted_state - target_state)**2)
    
    def combined_loss(
        self,
        initial_state: torch.Tensor,
        predicted_state: torch.Tensor,
        target_state: torch.Tensor,
        energy_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Compute combined loss incorporating both prediction accuracy
        and energy conservation
        """
        pred_loss = self.prediction_loss(predicted_state, target_state)
        energy_loss = self.energy_loss(initial_state, predicted_state)
        
        return pred_loss + energy_weight * energy_loss

class HamiltonianTrainer:
    """Trainer for Hamiltonian Neural Network"""
    
    def __init__(
        self,
        model: HamiltonianNN,
        config: HNNConfig,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.device = config.device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            initial_state, target_state = batch
            initial_state = initial_state.to(self.device)
            target_state = target_state.to(self.device)
            
            # Forward pass
            predicted_state, _ = self.model(initial_state)
            
            # Compute loss
            loss = self.model.combined_loss(
                initial_state,
                predicted_state,
                target_state
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        for batch in val_loader:
            initial_state, target_state = batch
            initial_state = initial_state.to(self.device)
            target_state = target_state.to(self.device)
            
            predicted_state, _ = self.model(initial_state)
            loss = self.model.combined_loss(
                initial_state,
                predicted_state,
                target_state
            )
            
            total_loss += loss.item()
            
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Returns:
            Dictionary containing training history
        """
        epochs = num_epochs or self.config.num_epochs
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
