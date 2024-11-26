# services/hamiltonian_agent/train.py

import argparse
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import wandb  # for experiment tracking

from models.hnn_model import HamiltonianNN, HamiltonianTrainer, HNNConfig
from data.data_preparation import MarketDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages training experiment lifecycle"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.save_dir = Path(self.config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self._init_experiment()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required configuration fields
        required_fields = [
            'data.symbols',
            'data.start_date',
            'data.end_date',
            'model.input_dim',
            'training.batch_size'
        ]
        
        for field in required_fields:
            if not self._check_config_field(config, field):
                raise ValueError(f"Missing required config field: {field}")
                
        return config

    def _check_config_field(self, config: Dict[str, Any], field: str) -> bool:
        """Check if nested config field exists"""
        parts = field.split('.')
        current = config
        for part in parts:
            if part not in current:
                return False
            current = current[part]
        return True

    def _init_experiment(self):
        """Initialize experiment tracking"""
        if self.config.get('monitoring', {}).get('use_wandb', False):
            wandb.init(
                project="hamiltonian-nn",
                config=self.config,
                name=f"hnn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def run(self):
        """Run training experiment"""
        try:
            # Initialize data
            logger.info("Preparing data...")
            data_loaders = self._prepare_data()
            
            # Initialize model
            logger.info("Initializing model...")
            model, trainer = self._initialize_model()
            
            # Train model
            logger.info("Starting training...")
            history = trainer.train(
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val']
            )
            
            # Save results
            logger.info("Saving results...")
            self._save_results(model, history)
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            if wandb.run:
                wandb.finish()

    def _prepare_data(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Prepare data loaders"""
        preprocessor = MarketDataPreprocessor(
            symbols=self.config['data']['symbols'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            sequence_length=self.config['data']['sequence_length'],
            train_split=self.config['data']['train_split'],
            batch_size=self.config['training']['batch_size']
        )
        
        train_loader, val_loader = preprocessor.prepare_data()
        
        return {
            'train': train_loader,
            'val': val_loader,
            'preprocessor': preprocessor
        }

    def _initialize_model(self) -> tuple[HamiltonianNN, HamiltonianTrainer]:
        """Initialize model and trainer"""
        model_config = HNNConfig(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            learning_rate=self.config['training']['learning_rate'],
            batch_size=self.config['training']['batch_size'],
            num_epochs=self.config['training']['num_epochs']
        )
        
        model = HamiltonianNN(model_config)
        trainer = HamiltonianTrainer(model=model, config=model_config)
        
        return model, trainer

    def _save_results(
        self,
        model: HamiltonianNN,
        history: Dict[str, Any]
    ):
        """Save model and training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.save_dir / f"hnn_model_{timestamp}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config.__dict__,
            'history': history
        }, model_path)
        
        # Save metrics
        metrics_path = self.save_dir / f"metrics_{timestamp}.yml"
        with open(metrics_path, 'w') as f:
            yaml.dump(history, f)
            
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='conf/training_config.yml',
        help='Path to training configuration file'
    )
    args = parser.parse_args()
    
    experiment = ExperimentManager(args.config)
    experiment.run()

if __name__ == "__main__":
    main()
