#!/usr/bin/env python3
"""
Convertit le modèle PyTorch entraîné pour l'API - Version corrigée
"""

import torch
import pickle
import os
from datetime import datetime
from CNN_Model import CatDogCNN
from model_wrapper import PyTorchModelWrapper
import numpy as np

def convert_model_for_api():
    print("🔄 CONVERSION MODÈLE POUR API (VERSION CORRIGÉE)")
    print("=" * 50)
    
    # 1. Vérifier que le modèle PyTorch existe
    if not os.path.exists('./best_cnn_model_pytorch.pth'):
        print("❌ Fichier best_cnn_model_pytorch.pth manquant!")
        print("⚠️ Avez-vous bien terminé l'entraînement ?")
        return False
    
    print("✅ Modèle PyTorch trouvé")
    
    # 2. Charger le modèle PyTorch
    try:
        checkpoint = torch.load('./best_cnn_model_pytorch.pth', map_location='cpu')
        print(f"📊 Epoch: {checkpoint.get('epoch', 'Unknown')}")
        val_acc = checkpoint.get('val_acc', 0)
        print(f"📈 Val Accuracy: {val_acc:.2f}%")
        
        # Créer le modèle et charger les poids
        model = CatDogCNN(num_classes=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Mode évaluation
        
        print("✅ Modèle PyTorch chargé et en mode évaluation")
        
    except Exception as e:
        print(f"❌ Erreur chargement modèle PyTorch: {e}")
        return False
    
    # 3. Créer le wrapper avec la classe externe
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    wrapper = PyTorchModelWrapper(model, device=device)
    
    # 4. Test rapide du wrapper
    print(f"\n🧪 Test rapide du wrapper (device: {device}):")
    try:
        # Test avec image aléatoire
        test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = wrapper.predict_proba(test_image)
        print(f"✅ Test réussi - Prédiction: {pred.flatten()[0]:.4f}")
        
        # Test avec image différente
        test_image2 = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred2 = wrapper.predict_proba(test_image2)
        print(f"✅ Test 2 réussi - Prédiction: {pred2.flatten()[0]:.4f}")
        
        # Vérifier que les prédictions sont différentes
        diff = abs(pred.flatten()[0] - pred2.flatten()[0])
        if diff > 0.001:
            print(f"✅ Les prédictions varient (diff: {diff:.4f}) - Modèle fonctionne!")
        else:
            print(f"⚠️ Prédictions très similaires (diff: {diff:.4f})")
            
    except Exception as e:
        print(f"❌ Erreur test wrapper: {e}")
        return False
    
    # 5. Sauvegarder pour l'API
    api_model_data = {
        'model': wrapper,
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'model_type': 'CNN_PyTorch',
        'device': device,
        'saved_at': datetime.now().isoformat(),
        'val_accuracy': val_acc,
        'epoch': checkpoint.get('epoch', 'Unknown')
    }
    
    try:
        with open('./best_cnn_model.pkl', 'wb') as f:
            pickle.dump(api_model_data, f)
        
        file_size = os.path.getsize('./best_cnn_model.pkl') / (1024*1024)
        print(f"\n✅ Modèle API sauvegardé: best_cnn_model.pkl ({file_size:.1f} MB)")
        
        # Afficher les stats du modèle
        print(f"📊 Statistiques du modèle:")
        print(f"   • Accuracy validation: {val_acc:.2f}%")
        print(f"   • Device: {device}")
        print(f"   • Epoch d'arrêt: {checkpoint.get('epoch', 'Unknown')}")
        
        if val_acc < 70:
            print("⚠️ ATTENTION: Accuracy < 70% - Le modèle pourrait être sous-entraîné")
        elif val_acc > 95:
            print("⚠️ ATTENTION: Accuracy > 95% - Possible overfitting")
        else:
            print("✅ Accuracy dans la plage normale (70-95%)")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde API: {e}")
        return False
    
    print("\n🎉 CONVERSION TERMINÉE!")
    print("▶️ Vous pouvez maintenant relancer l'API:")
    print("   python api.py")
    print("\n📝 Le modèle devrait maintenant donner des prédictions variables!")
    
    return True

if __name__ == "__main__":
    success = convert_model_for_api()
    if success:
        print("\n✅ SUCCÈS - Modèle prêt pour l'API")
    else:
        print("\n❌ ÉCHEC - Vérifiez les erreurs ci-dessus")