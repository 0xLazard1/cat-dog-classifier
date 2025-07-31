#!/usr/bin/env python3
"""
API FastAPI pour la classification d'images chats/chiens
VERSION PYTORCH UNIQUEMENT - Optimisé Apple Silicon M4
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime
import uvicorn
import pickle
import torch

# Configuration de l'API
app = FastAPI(
    title="API Classification Chats/Chiens - PyTorch M4", 
    version="4.0.0",
    description="Classification d'images avec CNN PyTorch optimisé Apple Silicon"
)

# Configuration CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model_wrapper = None
model_loaded = False

class PyTorchModelManager:
    """Gestionnaire pour modèle PyTorch uniquement"""
    
    def __init__(self):
        self.model = None
        self.model_type = "CNN_PyTorch_M4"
        self.input_shape = (224, 224, 3)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def load_model(self):
        """Charge le modèle PyTorch"""
        model_paths = [
            './best_cnn_model.pkl',
            './models/best_cnn_model.pkl'
        ]
        
        try:
            # Charger le modèle PyTorch
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            data = pickle.load(f)
                        
                        self.model = data['model']
                        print(f"✅ Modèle PyTorch chargé depuis {path}")
                        print(f"🚀 Device: {self.device}")
                        return True
                        
                    except Exception as e:
                        print(f"⚠️ Erreur chargement {path}: {e}")
                        continue
            
            # Si aucun modèle trouvé, créer un modèle de test
            print("⚠️ Aucun modèle trouvé, création d'un modèle PyTorch de test")
            from CNN_Model import CatDogCNN
            pytorch_model = CatDogCNN()
            
            # Créer le wrapper
            class PyTorchModelWrapper:
                def __init__(self, model):
                    self.model = model
                    self.input_shape = (224, 224, 3)
                    self.num_classes = 1
                    
                def predict_proba(self, x):
                    return self.model.predict_proba(x)
            
            self.model = PyTorchModelWrapper(pytorch_model)
            print("🧪 Modèle PyTorch de test créé")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Prétraite une image pour PyTorch CNN"""
        # Conversion RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionner
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convertir en array et normaliser
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Ajouter dimension batch: (1, 224, 224, 3)
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image: Image.Image) -> dict:
        """Fait une prédiction avec PyTorch CNN"""
        if not self.model:
            raise ValueError("Modèle non chargé")
        
        # Preprocessing
        image_processed = self.preprocess_image(image)
        
        # Prédiction
        try:
            probabilities = self.model.predict_proba(image_processed)
            
            # Extraire la probabilité
            if isinstance(probabilities, np.ndarray):
                prob_dog = float(probabilities.flatten()[0])
            else:
                prob_dog = float(probabilities)
            
            # Assurer [0, 1]
            prob_dog = max(0.0, min(1.0, prob_dog))
            prob_cat = 1.0 - prob_dog
            
            # Classification
            predicted_class = "chien" if prob_dog > 0.5 else "chat"
            confidence = prob_dog if predicted_class == "chien" else prob_cat
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    "chat": float(prob_cat),
                    "chien": float(prob_dog)
                },
                "model_type": self.model_type,
                "device": str(self.device)
            }
            
        except Exception as e:
            raise ValueError(f"Erreur de prédiction: {str(e)}")

# Initialiser le gestionnaire
model_manager = PyTorchModelManager()

@app.on_event("startup")
async def startup_event():
    """Démarrage - charge le modèle PyTorch"""
    global model_loaded
    print("🚀 Démarrage API PyTorch...")
    print(f"🔧 PyTorch version: {torch.__version__}")
    model_loaded = model_manager.load_model()
    if model_loaded:
        print("✅ API PyTorch prête (Apple Silicon optimized)")
    else:
        print("⚠️ API démarrée avec modèle de test")

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "🐱🐶 API Classification Chats/Chiens",
        "version": "4.0.0",
        "model": "CNN PyTorch (Apple Silicon M4)",
        "status": "running",
        "model_loaded": model_loaded,
        "device": str(model_manager.device),
        "pytorch_version": torch.__version__,
        "input_size": "224x224",
        "expected_accuracy": "85-95%"
    }

@app.get("/health")
async def health():
    """État de santé de l'API"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": model_manager.model_type,
        "device": str(model_manager.device),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """Informations détaillées sur le modèle"""
    return {
        "architecture": "CNN PyTorch (5 Conv + 2 Dense)",
        "input_shape": model_manager.input_shape,
        "model_type": model_manager.model_type,
        "device": str(model_manager.device),
        "preprocessing": "Standard normalization",
        "expected_accuracy": "85-95%",
        "optimized_for": "Apple Silicon M4"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Classification d'une image avec PyTorch CNN"""
    
    # Vérifications
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Charger l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prédiction
        result = model_manager.predict(image)
        
        # Métadonnées
        result.update({
            "filename": file.filename,
            "file_size": len(contents),
            "timestamp": datetime.now().isoformat(),
            "api_version": "4.0.0"
        })
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

if __name__ == "__main__":
    print("🐱🐶 API Classification PyTorch - Apple Silicon M4")
    print("📡 URL: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)