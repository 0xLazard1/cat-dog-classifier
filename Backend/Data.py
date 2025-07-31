import os 
import numpy as np
from skimage import io, transform
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class DataLoading:
    """Classe avancée pour le chargement et préprocessing d'images"""
    
    def __init__(self, filename: str, image_size=(224, 224), use_advanced=True):
        self.filename = filename
        self.image_size = image_size  # Augmenté à 224x224 pour CNN
        self.use_advanced = use_advanced
    
    def preprocess_image(self, image_path, augment=False):
        """Prétraite une image individuelle avec preprocessing avancé"""
        try:
            if self.use_advanced:
                return self._preprocess_advanced(image_path, augment)
            else:
                return self._preprocess_legacy(image_path, augment)
                
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {image_path}: {e}")
            return None
    
    def _preprocess_advanced(self, image_path, augment=False):
        """Preprocessing avancé pour CNN avec normalisation ImageNet"""
        # Charger avec PIL pour meilleure qualité
        image = Image.open(image_path).convert('RGB')
        
        # Redimensionner avec antialiasing
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convertir en array
        image = np.array(image, dtype=np.float32)
        
        # Normalisation ImageNet
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        image = image / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Augmentation de données si demandée
        if augment:
            image = self._apply_augmentation(image)
        
        return image
    
    def _preprocess_legacy(self, image_path, augment=False):
        """Preprocessing legacy pour compatibilité"""
        # Charger l'image
        image = io.imread(image_path)
        
        # Convertir en RGB si nécessaire
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Redimensionner
        image = transform.resize(image, self.image_size, preserve_range=True)
        
        # Normaliser entre 0 et 1
        image = image / 255.0
        
        return image.astype(np.float32)
    
    def _apply_augmentation(self, image):
        """Applique des augmentations aléatoires"""
        # Rotation aléatoire
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = self._rotate_image(image, angle)
        
        # Flip horizontal
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Ajustement de luminosité
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, -2.5, 2.5)  # Limites pour ImageNet normalization
        
        return image
    
    def _rotate_image(self, image, angle):
        """Rotation d'image avec remplissage"""
        # Dénormaliser temporairement pour OpenCV
        temp_image = ((image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])) * 255.0
        temp_image = np.clip(temp_image, 0, 255).astype(np.uint8)
        
        # Rotation
        h, w = temp_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(temp_image, matrix, (w, h), borderValue=(128, 128, 128))
        
        # Renormaliser
        rotated = rotated.astype(np.float32) / 255.0
        rotated = (rotated - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        return rotated
    
    def path_creation(self):
        """Crée la liste des chemins vers les dossiers de données"""
        all_paths = []
        sub_folders = ['test', 'train']
        classes = ['cats', 'dogs']
        
        for sub in sub_folders:
            for class_name in classes:
                path = os.path.join(self.filename, sub, class_name)
                if os.path.exists(path):
                    all_paths.append(path)
                else:
                    print(f"⚠️ Attention: Le chemin {path} n'existe pas")
        
        return all_paths