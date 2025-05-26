"""
Fichier: main.py
But: API FastAPI pour la prédiction de segmentation d'images
    en utilisant un modèle EfficientNet (unet_efficientnetb3)
    entraîné et sauvegardé, ainsi que les fonctions du fichier fonctions.py.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import os

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calcule le Dice Coefficient.
    Correction : conversion explicite en float32.
    """
    y_true = tf.keras.backend.cast(y_true, "float32")
    y_pred = tf.keras.backend.cast(y_pred, "float32")
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# 1) Import de segmentation_models et du fichier fonctions.py
import segmentation_models as sm

# Important : préciser à segmentation_models qu'on travaille avec tf.keras
sm.set_framework("tf.keras")

# 2) Définir les objets personnalisés nécessaires pour recharger le modèle
#    (perte, métriques, etc.), en pointant vers segmentation_models et fonctions.py
custom_objects = {
    # Perte utilisée pour l’entraînement
    "categorical_focal_jaccard_loss": sm.losses.categorical_focal_jaccard_loss,

    # Métriques utilisées : IoU score, F1-Score
    "iou_score": sm.metrics.iou_score,
    "f1-score": sm.metrics.FScore(),  # ou "FScore": sm.metrics.FScore()

    # Fonction dice_coef définie dans fonctions.py
    "dice_coef": dice_coef
}

# 3) Chargement du modèle entraîné
modele_path = "best_model.keras"  # EfficientNet
model = tf.keras.models.load_model(
    modele_path,
    compile=False,  # On évite la recompilation si on ne poursuit pas l'entraînement
    custom_objects=custom_objects
)

# 4) Création de l'application FastAPI
app = FastAPI()

@app.get("/")
async def root():
    """
    Endpoint racine : Premier lancement 01/03/2025
    """
    return {"message": "Bienvenue sur l'API de segmentation (EfficientNet) Future Vision Transport"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """
    Endpoint POST /predict/
    - Reçoit une image en entrée (via UploadFile).
    - Prétraitement : conversion en RGB, resize (256,256), normalisation.
    - Passage au réseau (model.predict).
    - Transformation du masque de probabilités en masques de classes (argmax).
    - Renvoi d’une image PNG (mask prédict) sous forme de flux binaire.
    """
    # 1) Charger et pré-traiter l'image
    img = Image.open(image.file).convert("RGB").resize((256, 256))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape = (1, 256, 256, 3)

    # 2) Prédiction du masque
    pred_mask = model.predict(img_array)[0]  # shape = (256, 256, 8)
    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)

    # 3) Convertir le masque en PNG
    #    On normalise sur 255 uniquement pour visualiser plus facilement
    if pred_mask.max() > 0:
        mask_img = Image.fromarray((pred_mask * (255 / pred_mask.max())).astype(np.uint8))
    else:
        # Si le masque est tout à zéro, on le retourne tel quel
        mask_img = Image.fromarray(pred_mask)

    # 4) Retour sous forme de StreamingResponse (image/png)
    mask_img_io = io.BytesIO()
    mask_img.save(mask_img_io, "PNG")
    mask_img_io.seek(0)

    return StreamingResponse(mask_img_io, media_type="image/png")
