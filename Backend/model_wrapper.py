#!/usr/bin/env python3
"""
Wrapper pour compatibilité API PyTorch
"""

import torch
import numpy as np

class PyTorchModelWrapper:
    """Wrapper pour rendre le modèle PyTorch compatible avec l'API"""
    
    def __init__(self, pytorch_model, device='cpu'):
        self.model = pytorch_model
        self.device = device
        self.input_shape = (224, 224, 3)
        self.num_classes = 1
        
        # S'assurer que le modèle est en mode évaluation
        self.model.eval()
        self.model.to(device)
        
    def predict_proba(self, x):
        """Compatible avec l'API existante"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                # Convertir numpy en tensor PyTorch
                x_tensor = torch.from_numpy(x).float()
                # Réorganiser les dimensions (B,H,W,C) -> (B,C,H,W)
                if x_tensor.dim() == 4 and x_tensor.shape[-1] == 3:
                    x_tensor = x_tensor.permute(0, 3, 1, 2)
            else:
                x_tensor = x
            
            # Envoyer sur le bon device
            x_tensor = x_tensor.to(self.device)
            
            # Prédiction
            outputs = self.model(x_tensor)
            return outputs.cpu().numpy()