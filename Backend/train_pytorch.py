#!/usr/bin/env python3
"""
Script d'entraînement PyTorch optimisé pour Apple Silicon M4
"""

import sys
import os

# Vérifier PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} détecté")
    if torch.backends.mps.is_available():
        print("🚀 GPU Apple Silicon (MPS) disponible!")
    else:
        print("💻 Mode CPU")
except ImportError:
    print("❌ PyTorch non installé!")
    print("Installez avec: pip install torch torchvision")
    sys.exit(1)

# Lancer l'entraînement
if __name__ == "__main__":
    from CNN_Model import main
    main()