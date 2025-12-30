import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models import initialize_model, ImagePreprocessor
from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, DEVICE, RANDOM_SEED,
    OUTPUTS_DIR, PROCESSED_DATA_DIR
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class PropertyDataset(Dataset):

    def __init__(self, tabular_data, image_paths, targets=None):
        self.tabular_data = torch.FloatTensor(tabular_data)
        self.image_paths = image_paths
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.image_preprocessor = ImagePreprocessor()

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        tabular = self.tabular_data[idx]
        image = self.image_preprocessor.load_image(self.image_paths[idx])

        if self.targets is not None:
            target = self.targets[idx]
            return image, tabular, target
        else:
            return image, tabular


def calculate_metrics(predictions, targets):
    predictions = predictions.flatten()
    targets = targets.flatten()

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    mape = np.mean(np.abs((targets - predictions) / targets)) * 100

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


class PropertyValueTrainer:

    def __init__(self, model, device, learning_rate=LEARNING_RATE):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for images, tabular, targets in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images, tabular)
            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, tabular, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                tabular = tabular.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images, tabular)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss

        return metrics

    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['learning_rate'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
                  f"R²: {val_metrics['r2']:.4f} | MAE: {val_metrics['mae']:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print("✓ Saved new best model!")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        self.plot_training_history()

    def save_checkpoint(self, filename='checkpoint.pth'):
        filepath = OUTPUTS_DIR / 'models' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)

    def load_checkpoint(self, filename='best_model.pth'):
        filepath = OUTPUTS_DIR / 'models' / filename
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {filepath}")

    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.history['val_rmse'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.history['val_r2'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Validation R² Score')
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()

        save_path = OUTPUTS_DIR / 'visualizations' / 'training_history.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved training history plot to {save_path}")
        plt.close()

def main():
    print("="*80)
    print("PROPERTY VALUATION - TRAINING PIPELINE")
    print("="*80)

    print("\nLoading processed data...")
    data = np.load(PROCESSED_DATA_DIR / 'train_processed.npz', allow_pickle=True)

    train_dataset = PropertyDataset(data['X_train'], data['images_train'], data['y_train'])
    val_dataset = PropertyDataset(data['X_val'], data['images_val'], data['y_val'])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    tabular_input_size = data['X_train'].shape[1]
    model, device = initialize_model(tabular_input_size, 'multimodal')

    trainer = PropertyValueTrainer(model, device)
    trainer.train(train_loader, val_loader)

    final_metrics = {
        'best_val_loss': float(trainer.best_val_loss),
        'final_r2': float(trainer.history['val_r2'][-1]),
        'final_rmse': float(trainer.history['val_rmse'][-1]),
        'total_epochs': len(trainer.history['train_loss']),
        'training_date': datetime.now().isoformat()
    }

    metrics_path = OUTPUTS_DIR / 'models' / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nSaved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()

