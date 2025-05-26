# 🚗 Projet 8 - Segmentation d'images pour véhicules autonomes

Ce projet est réalisé dans le cadre du parcours *AI Engineer* d’OpenClassrooms, en partenariat avec **Future Vision Transport** (cas d'usage fictif).  
L'objectif est de développer un **modèle de segmentation sémantique d’images urbaines** à partir du dataset **Cityscapes**, et de fournir une **API de prédiction** ainsi qu’une **interface web de démonstration**.

---

## 🎯 Objectifs du projet

1. **Préparer les données** (remapping des masques, générateur personnalisé, one-hot encoding)
2. **Entraîner différents modèles de segmentation** sur les 8 classes principales de Cityscapes.
3. **Comparer les performances** avec plusieurs métriques (IoU, Dice, Accuracy).
4. **Concevoir une API FastAPI** capable de retourner un masque prédit à partir d’une image.
5. **Créer une application Streamlit** simple et interactive pour tester l’API.

---

## 📂 Arborescence du dépôt

```
├── API.py                        → API FastAPI pour prédiction
├── application_Streamlit.py     → Application Streamlit (test + affichage masques)
├── fonctions.py                 → Fonctions communes (données, modèles, métriques, etc.)
├── note_technique.pdf           → Présentation complète du projet (PDF)
├── scripts.ipynb                → Notebook de traitement et entraînement
└── README.md                    → Présentation du projet
```

---

## 🧠 Modèles testés

| Modèle          | Backbone        | Particularité                              |
|------------------|------------------|--------------------------------------------|
| Unet Mini        | Aucun            | Léger, rapide à entraîner                  |
| Unet VGG16       | VGG16            | Précis, bon compromis                      |
| Unet ResNet50    | ResNet50         | Performant, stable                         |
| Unet EfficientNetB3 | EfficientNetB3 | Le plus performant (IoU et Dice)           |

---

## 📊 Métriques utilisées

- **IoU Score** (Intersection over Union)
- **Dice Coefficient** (F1 score pixel-level)
- **Accuracy**
- **Loss (CrossEntropy / Focal Jaccard Loss)**

---

## 🧪 Déploiement & Test

### API (FastAPI) sur AWS

- Entrée : image RGB `.png`, `.jpg` ou `.jpeg`
- Sortie : masque segmenté sous forme d’image PNG (grayscale, 0–7)
- Lien API : `/predict/`  
- Exemple de modèle utilisé : `unet_efficientnetb3` sauvegardé en `.keras`

### Interface utilisateur (Streamlit)

- Upload d'une image et affichage du **masque prédit**.
- Upload optionnel du masque réel pour comparaison.
- **Colorisation** des 8 classes avec légende.

---

## 📈 Résultats

- Meilleures performances atteintes avec `EfficientNetB3`
- Entraînement réalisé sur 2500 images pour respecter les contraintes matérielles
- Score Dice & IoU satisfaisants pour un usage en environnement urbain

---

## 🧠 Auteur

Projet réalisé par **AnthonyJVID**, dans le cadre du parcours *AI Engineer* chez OpenClassrooms.

---

## 📄 Licence

Projet à visée pédagogique. Dataset Cityscapes © utilisé dans un but non commercial.
