# 🐱🐶 Classification Chats/Chiens - PyTorch CNN

API FastAPI avec CNN PyTorch optimisé pour Apple Silicon (M4) pour classifier des images de chats et chiens.

## 🎯 **Performances**
- **Accuracy**: 87.59% (validation)
- **Architecture**: CNN personnalisé (5 Conv + 2 Dense)
- **Optimisation**: Apple Silicon MPS (GPU M4)
- **Temps prédiction**: ~100-200ms

## 🚀 **Installation rapide**

```bash
# 1. Installer PyTorch pour Apple Silicon
pip install torch torchvision torchaudio

# 2. Installer autres dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle (si pas déjà fait)
python train_pytorch.py

# 4. Configurer le modèle pour l'API (si pas déjà fait)
python setup_model.py

# 5. Lancer l'API
python api.py
```

## 📡 **Endpoints API**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations API + statistiques |
| `/health` | GET | État santé + device utilisé |
| `/model-info` | GET | Architecture détaillée du modèle |
| `/predict` | POST | Classification d'image |

## 📚 **Documentation interactive**

Une fois l'API démarrée : **http://localhost:8000/docs**

## 🧪 **Test rapide**

```bash
# Test avec curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@votre_image.jpg"

# Réponse exemple
{
  "predicted_class": "chien",
  "confidence": 0.92,
  "probabilities": {
    "chat": 0.08,
    "chien": 0.92
  },
  "model_type": "CNN_PyTorch_M4",
  "device": "mps"
}
```

## 🏗️ **Architecture technique**

### **Modèle CNN**
```python
# 5 blocs convolutionnels
Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Dropout
Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout  
Conv2D(128) -> BatchNorm -> ReLU -> MaxPool -> Dropout
Conv2D(256) -> BatchNorm -> ReLU -> MaxPool -> Dropout
Conv2D(512) -> BatchNorm -> ReLU -> AdaptiveAvgPool

# 2 couches denses
Dense(512) -> BatchNorm -> ReLU -> Dropout
Dense(256) -> ReLU -> Dropout  
Dense(1) -> Sigmoid
```

### **Optimisations Apple Silicon**
- ✅ **MPS Backend**: Utilisation MPS MACBOOK PRO M4
- ✅ **Memory Pinning**: Transferts CPU↔GPU optimisés
- ✅ **Batch Processing**: Support images multiples
- ✅ **Mixed Precision**: Float32 optimisé

## 📁 **Structure projet**

```
Backend/
├── 📄 api.py                 # API FastAPI principale
├── 🧠 CNN_Model.py           # Architecture CNN PyTorch
├── 🔧 model_wrapper.py       # Wrapper compatibilité API
├── 🏋️ train_pytorch.py       # Script d'entraînement
├── ⚙️ setup_model.py         # Configuration modèle pour API
├── 📊 Data.py               # Preprocessing images
├── 📋 requirements.txt      # Dépendances Python
├── 💾 best_cnn_model.pkl    # Modèle entraîné (API)
├── 💾 best_cnn_model_pytorch.pth  # Checkpoint PyTorch
└── 📂 data/                 # Dataset
    ├── train/cats/          # Images chats (entraînement)
    ├── train/dogs/          # Images chiens (entraînement)  
    ├── test/cats/           # Images chats (test)
    └── test/dogs/           # Images chiens (test)
```

## 🎛️ **Configuration avancée**

### **Variables d'environnement**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Optimisation mémoire M4
export OMP_NUM_THREADS=8                      # Threads CPU
```

### **Paramètres modèle**
- **Input Size**: 224×224×3
- **Batch Size**: 64 (entraînement)
- **Learning Rate**: 0.001 (Adam)
- **Early Stopping**: 8 epochs patience
- **Data Augmentation**: Rotation, flip, brightness

## 🔧 **Maintenance**

### **Ré-entraîner le modèle**
```bash
# Si vous voulez améliorer les performances
python train_pytorch.py

# Puis reconfigurer pour l'API
python setup_model.py
```

### **Monitoring**
```bash
# Vérifier l'état de l'API
curl http://localhost:8000/health

# Voir les métriques détaillées
curl http://localhost:8000/model-info
```

## 🚨 **Troubleshooting**

| Problème | Solution |
|----------|----------|
| `best_cnn_model.pkl manquant` | Lancez `python setup_model.py` |
| `MPS not available` | Modèle fonctionne sur CPU (plus lent) |
| `Prédictions identiques` | Relancez `python setup_model.py` |
| `Memory error` | Réduisez batch_size dans train_pytorch.py |

## 📈 **Métriques de développement**

- **Temps d'entraînement**: ~15-30 min (M4 GPU)
- **Taille modèle**: ~100-200 MB
- **RAM requise**: ~1-2 GB (API + modèle)
- **Throughput**: ~50-100 images/seconde

## 🎯 **Prochaines étapes**

1. **Améliorer accuracy**: Transfer learning (ResNet, EfficientNet)
2. **Optimiser vitesse**: Quantization, ONNX export
3. **Étendre classes**: Plus d'animaux domestiques
4. **Deploy cloud**: Docker + Cloud Run / Railway
5. **Monitoring**: Logging, métriques Prometheus

---

**🚀 Développé avec PyTorch + FastAPI**
