#!/usr/bin/env python3
"""
Modèle CNN PyTorch optimisé pour Apple Silicon (M4)
Utilise Metal Performance Shaders (MPS) pour accélération GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

class CatDogCNN(nn.Module):
    """CNN PyTorch pour classification chats/chiens"""
    
    def __init__(self, num_classes=1):
        super(CatDogCNN, self).__init__()
        
        # Architecture CNN
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Bloc 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Bloc 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classificateur
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        self.input_shape = (224, 224, 3)
        self.num_classes = num_classes
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x):
        """Compatible avec l'API existante"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                # Convertir numpy en tensor PyTorch
                x = torch.from_numpy(x).float()
                # Réorganiser les dimensions si nécessaire (H,W,C) -> (C,H,W)
                if x.dim() == 4 and x.shape[-1] == 3:
                    x = x.permute(0, 3, 1, 2)
            
            # Assurer qu'on est sur le bon device
            device = next(self.parameters()).device
            x = x.to(device)
            
            outputs = self.forward(x)
            return outputs.cpu().numpy()


class CatDogDataset:
    """Gestion des données avec transformations PyTorch"""
    
    @staticmethod
    def get_transforms(is_train=True):
        """Transformations avec augmentation pour l'entraînement"""
        if is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def create_loaders(train_dir, batch_size=64, num_workers=4):
        """Crée les DataLoaders PyTorch"""
        # Créer les datasets
        full_dataset = datasets.ImageFolder(
            train_dir,
            transform=CatDogDataset.get_transforms(is_train=True)
        )
        
        # Split train/validation (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Appliquer les transformations de validation
        val_dataset.dataset.transform = CatDogDataset.get_transforms(is_train=False)
        
        # Créer les loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"📊 Données d'entraînement: {len(train_dataset)} images")
        print(f"📊 Données de validation: {len(val_dataset)} images")
        
        return train_loader, val_loader


class Trainer:
    """Gestionnaire d'entraînement PyTorch optimisé pour Apple Silicon"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, train_loader, optimizer):
        """Entraîne une epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour de la barre de progression
            accuracy = 100 * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Valide le modèle"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Entraînement complet avec early stopping"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_acc = 0
        patience = 8
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            # Entraînement
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler
            scheduler.step(val_loss)
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, 'best_cnn_model_pytorch.pth')
                print(f"✅ Nouveau meilleur modèle sauvegardé (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping après {epoch+1} epochs")
                    break
        
        print(f"\n🎯 Meilleure accuracy: {best_val_acc:.2f}%")
        return self.history


def save_model_for_api(model, filepath='best_cnn_model.pkl'):
    """Sauvegarde le modèle pour l'API"""
    # Créer un wrapper compatible avec l'API existante
    class PyTorchModelWrapper:
        def __init__(self, pytorch_model):
            self.model = pytorch_model
            self.input_shape = (224, 224, 3)
            self.num_classes = 1
            
        def predict_proba(self, x):
            return self.model.predict_proba(x)
    
    wrapper = PyTorchModelWrapper(model)
    
    # Sauvegarder
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': wrapper,
            'input_shape': (224, 224, 3),
            'num_classes': 1,
            'model_type': 'CNN_PyTorch',
            'saved_at': datetime.now().isoformat()
        }, f)
    
    print(f"✅ Modèle sauvegardé pour l'API: {filepath}")


def main():
    """Fonction principale d'entraînement"""
    print("🐱🐶 ENTRAÎNEMENT CNN PYTORCH - APPLE SILICON OPTIMIZED")
    print("=" * 60)
    
    # Configuration device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Utilisation du GPU Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Utilisation du CPU")
    
    print(f"🔧 PyTorch version: {torch.__version__}")
    print("-" * 60)
    
    # Créer le modèle
    model = CatDogCNN()
    print(f"📊 Paramètres totaux: {sum(p.numel() for p in model.parameters()):,}")
    
    # Créer les loaders
    train_loader, val_loader = CatDogDataset.create_loaders(
        './data/train',
        batch_size=64,
        num_workers=4  # Optimisé pour M4
    )
    
    # Entraîner
    trainer = Trainer(model, device)
    history = trainer.train(train_loader, val_loader, epochs=50, lr=0.001)
    
    # Sauvegarder pour l'API
    model.load_state_dict(torch.load('best_cnn_model_pytorch.pth')['model_state_dict'])
    save_model_for_api(model)
    
    print("\n✅ ENTRAÎNEMENT TERMINÉ!")


if __name__ == "__main__":
    main()