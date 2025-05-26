"""
app.py - Application Streamlit pour tester l'API de segmentation
Critères d'évaluation couverts :
- Présentation de l'image originale
- Envoi de l'image à l'API
- Récupération et affichage du masque prédit
- Affichage du masque réel (optionnel) pour comparaison
- Coloration des 8 classes
- Interface simple pour tester la prédiction
"""

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

# Définition des 8 classes (CLASSES_UTILES) et d'une palette de couleurs distinctes
CLASSES_UTILES = {
    "void": 0,
    "flat": 1,
    "construction": 2,
    "object": 3,
    "nature": 4,
    "sky": 5,
    "human": 6,
    "vehicle": 7
}

# Palette de couleurs (R, G, B) pour chaque classe
LABEL_COLORS = {
    0: (0, 0, 0),         # void: noir
    1: (128, 64, 128),    # flat
    2: (70, 70, 70),      # construction
    3: (190, 153, 153),   # object
    4: (107, 142, 35),    # nature
    5: (70, 130, 180),    # sky
    6: (220, 20, 60),     # human
    7: (119, 11, 32)      # vehicle
}

def colorize_mask(mask_img: Image.Image) -> Image.Image:
    """
    Convertit une image en niveaux de gris [0..255] en un masque coloré
    selon la palette LABEL_COLORS et les classes [0..7].
    On suppose que le masque retourné par l'API a été rescalé pour la visualisation.
    """
    mask_np = np.array(mask_img)  # shape (H, W), np.uint8
    if mask_np.max() > 0:
        # Convertir l'échelle pour obtenir des indices entiers 0..7
        classes_np = (mask_np / (mask_np.max() / 7.0)).round().astype(np.uint8)
    else:
        classes_np = mask_np

    # Créer une image couleur à partir des indices
    h, w = classes_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in LABEL_COLORS.items():
        color_img[classes_np == class_idx] = color
    return Image.fromarray(color_img)

# Configuration Streamlit
st.title("Application Web - Segmentation sur 8 classes")
st.write("Cette interface teste l'API de segmentation (modèle EfficientNet).")
st.write("Veuillez d'abord lancer l'API FastAPI sur le port 8000.")

# Upload de l'image originale
original_file = st.file_uploader("Choisissez l'image originale à segmenter", type=["png", "jpg", "jpeg"])
# Upload du masque réel (optionnel)
ground_truth_file = st.file_uploader("Choisissez le masque réel (optionnel)", type=["png", "jpg", "jpeg"])

if original_file is not None:
    # Charger et afficher l'image originale
    original_image = Image.open(original_file).convert("RGB")

    # Disposition en 3 colonnes
    cols = st.columns(3)

    with cols[0]:
        st.image(original_image, caption="Image originale", use_column_width=True)

    # Si le masque réel est fourni, l'afficher
    if ground_truth_file is not None:
        real_mask = Image.open(ground_truth_file)
        with cols[1]:
            st.image(real_mask, caption="Masque réel", use_column_width=True)
    else:
        with cols[1]:
            st.write("Masque réel non fourni")

    # Appel à l'API pour obtenir le masque prédit
    api_url = "http://35.180.51.112:8000/predict"  # URL de l'API sur AWS
    files = {"image": original_file.getvalue()}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        mask_pred = Image.open(io.BytesIO(response.content))
        mask_pred_color = colorize_mask(mask_pred)
        with cols[2]:
            st.image(mask_pred_color, caption="Masque prédit (colorisé)", use_column_width=True)
    else:
        st.error(f"Erreur lors de la récupération du masque prédit (code: {response.status_code})")

# Affichage de la légende des couleurs
st.subheader("Légende des 8 classes")
legend_cols = st.columns(4)
for i, label_name in enumerate(list(CLASSES_UTILES.keys())):
    class_id = CLASSES_UTILES[label_name]
    color = LABEL_COLORS[class_id]
    color_img = Image.new("RGB", (60, 30), color)
    col = legend_cols[i % 4]
    col.image(color_img, caption=f"{label_name} (id={class_id})", use_column_width=True)

st.write("Téléchargez une image et, éventuellement, le masque réel pour tester la segmentation.")