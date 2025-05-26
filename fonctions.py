# fonctions.py

# Importations nécessaires
import os
import tensorflow as tf
from cityscapesscripts.helpers.labels import name2label
from cityscapesscripts.preparation.json2labelImg import json2labelImg
import json
import numpy as np
import albumentations as A
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from albumentations import Compose, HorizontalFlip, Rotate, OneOf, RandomScale, Blur, GaussNoise, Resize
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, Resizing, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tqdm import tqdm
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from cityscapesscripts.helpers.labels import trainId2label
import time
import segmentation_models as sm
import pandas as pd
from pathlib import Path
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import glob


# Définition des classes utiles
CLASSES_UTILES = {
    "void": 0, "flat": 1, "construction": 2, "object": 3,
    "nature": 4, "sky": 5, "human": 6, "vehicle": 7
}

root_path = Path("C:/Users/icema/OpenClassrooms/AI_Engineer/venv8")
data_path = root_path / "data"
cityscapes_scripts_path = root_path / "notebooks/cityscapesScripts/cityscapesscripts"
images_path = data_path / "leftImg8bit"
masks_path = data_path / "gtFine"

def remap_classes(mask: np.ndarray) -> np.ndarray:
    """
    Convertit les classes Cityscapes originales (0-33) vers les 8 catégories principales définies.

    Retourne un masque avec uniquement des valeurs de 0 à 7.
    """
    # Définition précise du mapping basé sur les "labelIds" Cityscapes originaux
    labelIds_to_main_classes = {
        0: 0,   # unlabeled → void
        1: 0,   # ego vehicle → void
        2: 0,   # rectification border → void
        3: 0,   # out of roi → void
        4: 0,   # static → void
        5: 0,   # dynamic → void
        6: 0,   # ground → void
        7: 1,   # road → flat
        8: 1,   # sidewalk → flat
        9: 0,   # parking → void
        10: 0,  # rail track → void
        11: 2,  # building → construction<
        12: 2,  # wall → construction
        13: 2,  # fence → construction
        14: 0,  # guard rail → void
        15: 0,  # bridge → void
        16: 0,  # tunnel → void
        17: 3,  # pole → object
        18: 3,  # polegroup → object
        19: 3,  # traffic light → object
        20: 3,  # traffic sign → object
        21: 4,  # vegetation → nature
        22: 4,  # terrain → nature
        23: 5,  # sky → sky
        24: 6,  # person → human
        25: 6,  # rider → human
        26: 7,  # car → vehicle
        27: 7,  # truck → vehicle
        28: 7,  # bus → vehicle
        29: 7,  # caravan → vehicle
        30: 7,  # trailer → vehicle
        31: 7,  # train → vehicle
        32: 7,  # motorcycle → vehicle
        33: 7   # bicycle → vehicle
    }

    # Appliquer le remapping précis
    remapped_mask = np.copy(mask)
    for original_class, new_class in labelIds_to_main_classes.items():
        remapped_mask[mask == original_class] = new_class

    # Afficher les classes finales (vérification temporaire)
    # print(f"✅ Valeurs uniques après remapping : {np.unique(remapped_mask)}")

    return remapped_mask.astype(np.uint8)

def view_folder(dossier):
    dossier = Path(dossier)
    for sous_dossier in dossier.iterdir():
        if sous_dossier.is_dir():
            print(f"|-- {sous_dossier.name}")
            for sous_sous_dossier in sous_dossier.iterdir():
                if sous_sous_dossier.is_dir():
                    print(f"    |-- {sous_sous_dossier.name}")

def load_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Charge et normalise une image entre 0 et 1."""
    img = load_img(path, target_size=target_size)
    return img_to_array(img).astype("float32") / 255.0

def load_mask(path: str, target_size: Tuple[int, int], mask_mode="labelIds") -> np.ndarray:
    """
    Charge, redimensionne et remappe un masque.
    Applique systématiquement le remapping vers les 8 classes principales.

    Args:
        path (str): Chemin vers le masque.
        target_size (Tuple[int, int]): Taille de sortie (hauteur, largeur).
        mask_mode (str): "labelIds" pour les masques Cityscapes originaux, "trainIds" sinon.

    Returns:
        np.ndarray: Masque avec valeurs de classe entre 0 et 7.
    """
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    mask = img_to_array(mask).astype("uint8").squeeze()

    # Toujours appliquer le remapping pour garantir 8 classes
    mask = remap_classes(mask)

    return mask

def one_hot_encode_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Encode un masque en One-Hot."""

    # Vérifier les valeurs uniques avant l'encodage
    unique_values = np.unique(mask)
    if np.any(unique_values >= num_classes):
        print(f"Attention : Certaines valeurs de masques dépassent {num_classes-1}: {unique_values}")
        mask = np.clip(mask, 0, num_classes - 1)

    return np.eye(num_classes, dtype=np.uint8)[mask]

def decode_mask(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque One-Hot en format indexé."""
    return np.argmax(mask, axis=-1)

def get_augmentations(image_size: Tuple[int, int]) -> Compose:
    """Définit les transformations Albumentations pour l'entraînement."""
    return Compose([
        HorizontalFlip(p=0.2),
        Rotate(limit=15, p=0.2),
        RandomScale(scale_limit=0.1, p=0.2),
        Resize(*image_size, interpolation=cv2.INTER_NEAREST)
    ])

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, image_size=(256, 256), batch_size=64, num_classes=8, # TEST avec 512x512, 1024x1024, 512x1024, 1024x512, 256x512 et 512x256
                shuffle=True, augmentation_ratio=1.0, use_cache=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation_ratio = augmentation_ratio
        self.use_cache = use_cache
        self.cache = {}  # Cache des masques transformés
        self.augmentation = get_augmentations(image_size)
        self.on_epoch_end()

    def __getitem__(self, index):
        start_time = time.time()
        start = index * self.batch_size
        end = start + self.batch_size
        batch_image_paths = self.image_paths[start:end]
        batch_mask_paths = self.mask_paths[start:end]

        batch_images, batch_masks = [], []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            img = load_image(img_path, self.image_size)

            if self.use_cache and mask_path in self.cache:
                mask = self.cache[mask_path]
            else:
                mask = load_mask(mask_path, self.image_size, mask_mode="trainIds")
                if self.use_cache:
                    self.cache[mask_path] = mask

            if np.random.rand() < self.augmentation_ratio:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            batch_images.append(img)
            batch_masks.append(one_hot_encode_mask(mask, self.num_classes))

        elapsed_time = time.time() - start_time
        # print(f"📊 Génération batch {index} en {elapsed_time:.2f}s")

        return np.stack(batch_images), np.stack(batch_masks)

    def __len__(self):
        """Renvoie le nombre total de batches par epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Mélange les données après chaque epoch si shuffle est activé."""
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        """Affiche correctement un lot d'images et de masques."""
        batch_images, batch_masks = self.__getitem__(0)
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(decode_mask(batch_masks[i]), cmap="inferno")
            axes[i, 1].set_title("Mask (decoded)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()


# Test du DataGenerator
if __name__ == "__main__":
    train_gen = DataGenerator(
        image_paths=train_input_img_paths,
        mask_paths=train_label_ids_img_paths,
        image_size=(256, 256),  # TEST avec 512x512
        batch_size=64,  # TEST: 8, 16 ou 32
        num_classes=8,
        shuffle=True,
        augmentation_ratio=0.5
    )

    train_gen.visualize_batch(num_images=3)

    def on_epoch_end(self) -> None:
        """Mélange les données après chaque epoch si shuffle est activé."""
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        """Affiche correctement un lot d'images et de masques."""
        batch_images, batch_masks = self.__getitem__(0)
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(decode_mask(batch_masks[i]), cmap="inferno")
            axes[i, 1].set_title("Mask (decoded)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

def iou_coef(y_true, y_pred, smooth=1e-6):
    """
    Calcule l'Intersection over Union (IoU).
    Correction : conversion explicite en float32.
    """
    y_true = tf.keras.backend.cast(y_true, "float32")
    y_pred = tf.keras.backend.cast(y_pred, "float32")
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

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

def weighted_mean_iou(y_true, y_pred, num_classes=8, penalty_class=5, penalty_weight=3):
    """
    Calcule le Mean IoU avec une pénalité sur une classe spécifique (ex: "person").
    - penalty_class : indice de la classe à pénaliser (par défaut 5 = "person")
    - penalty_weight : coefficient de pondération pour cette classe
    """
    iou_values = []
    for i in range(num_classes):
        iou = iou_coef(y_true[..., i], y_pred[..., i])
        if i == penalty_class:
            iou *= penalty_weight  # Appliquer la pondération sur la classe "person"
        iou_values.append(iou)
    return tf.keras.backend.mean(tf.keras.backend.stack(iou_values))

def weighted_categorical_crossentropy(weights):
    """
    Fonction de perte pondérée pour accorder une pénalité plus élevée à certaines classes.

    Args:
        weights (np.array): Tableau des poids pour chaque classe.

    Returns:
        fonction perte utilisable avec model.compile()
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_sum(weights * y_true * tf.math.log(y_pred), axis=-1)
    return loss

    return loss

def mean_iou(y_true, y_pred, num_classes=8):
    """
    Fonction de métrique Mean IoU pour la segmentation multi-classes.
    """
    iou_values = []
    for i in range(num_classes):
        intersection = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        union = tf.reduce_sum(y_true[..., i]) + tf.reduce_sum(y_pred[..., i]) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)  # Ajout d'un epsilon pour éviter la division par 0
        iou_values.append(iou)
    return tf.reduce_mean(iou_values)

def unet_mini(input_shape=(256, 256, 3), num_classes=8):
    """
    Modèle U-Net simplifié corrigé pour la segmentation d'images multi-classes.
    Retourne le modèle U-Net mini compilé.
    """
    inputs = Input(shape=input_shape)

    # Encodeur
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)  # 128x128

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Bottleneck (64x64 si input est 256x256)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.3)(conv3)  # Dropout pour éviter l'overfitting (0.4 a 0.6)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)

    # Décodeur
    up4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv3)  # 64x64 vers 128x128
    up4 = Concatenate()([up4, conv2])
    conv4 = Conv2D(64, (3, 3), activation="relu", padding="same")(up4)
    conv4 = BatchNormalization()(conv4)

    up5 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv4)  # 128x128 → 256x256
    up5 = Concatenate()([up5, conv1])
    conv5 = Conv2D(32, (3, 3), activation="relu", padding="same")(up5)
    conv5 = BatchNormalization()(conv5)

    # Couche de sortie Softmax
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(conv5)

    # Création du modèle
    model = Model(inputs, outputs=outputs, name="unet_mini")

    return model

def get_logger(nom_modele:str):
    """
    Crée un CSVLogger pour enregistrer les métriques d'entraînement dans un fichier horodaté.

    Args:
        nom_modele (str): Nom du modèle utilisé pour le nom du fichier.

    Returns:
        CSVLogger: callback pour enregistrer les logs.
    """
    logs_dir = Path("test_modele")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{nom_modele}_{timestamp}.csv"

    return CSVLogger(log_filename, separator=",", append=False)


def entrainer_modele(model,
                    train_generator,
                    val_generator,
                    epochs:int = 50,
                    patience:int = 5):
    """
    Entraîne le modèle avec arrêt anticipé, ajustement dynamique du learning rate, sauvegarde du meilleur modèle, 
    sauvegarde automatique des métriques et des learning curves.

    Args:
        model (Model): Modèle Keras à entraîner.
        train_generator (Sequence): Générateur pour l'entraînement.
        val_generator (Sequence): Générateur pour la validation.
        epochs (int): Nombre maximal d'époques.
        patience (int): Nombre d'époques sans amélioration avant arrêt anticipé.

    Returns:
        history : L'historique d'entraînement Keras.
    """

    logs_dir = Path("test_modele")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = logs_dir / f"{model.name}_{timestamp}"

    # CSV Logger
    csv_logger = CSVLogger(f"{base_filename}.csv", separator=",", append=False)

    # Callback d'arrêt anticipé
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)

    # Callback de réduction du taux d'apprentissage
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience//2, verbose=1)

    # Callback de sauvegarde du meilleur modèle
    checkpoint = ModelCheckpoint(
        filepath=f"{base_filename}_best_model.keras",
        monitor="val_mean_iou",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    debut_entrainement = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[csv_logger, early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    temps_total = time.time() - debut_entrainement

    # Enregistrer le temps total d'entraînement
    df_metrics = pd.read_csv(f"{base_filename}.csv")
    df_metrics["temps_total_sec"] = temps_total
    df_metrics.to_csv(f"{base_filename}.csv", index=False)

    # Génération des learning curves
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mean_iou"], label="train_mean_iou")
    plt.plot(history.history["val_mean_iou"], label="val_mean_iou")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IoU")
    plt.title("Learning Curve (Mean IoU)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{base_filename}_learning_curves.png")
    plt.close()

    print(f"✅ Entraînement terminé en {temps_total:.2f} secondes.")
    print(f"📊 Métriques enregistrées dans : {base_filename}.csv")
    print(f"📈 Learning curves enregistrées dans : {base_filename}_learning_curves.png")
    print(f"📌 Meilleur modèle sauvegardé : {base_filename}_best_model.keras")

    return history

def unet_vgg16(input_shape=(256, 256, 3), num_classes=8, learning_rate=1e-4):
    """
    Modèle U-Net avec backbone VGG16 pré-entraîné sur ImageNet.
    """
    sm.set_framework("tf.keras")

    model = sm.Unet(
        backbone_name="vgg16",
        input_shape=input_shape,
        encoder_weights="imagenet",
        classes=num_classes,
        activation="softmax"
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[mean_iou, dice_coef, "accuracy"]
    )

    model._name = "unet_vgg16"
    return model

def unet_resnet50(input_shape=(256, 256, 3), num_classes=8, learning_rate=1e-4):
    """
    Modèle U-Net avec backbone ResNet50 pré-entraîné sur ImageNet.
    """
    sm.set_framework("tf.keras")

    model = sm.Unet(
        backbone_name="resnet50",
        input_shape=input_shape,
        encoder_weights="imagenet",
        classes=num_classes,
        activation="softmax"
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[mean_iou, dice_coef, "accuracy"]
    )

    model._name = "unet_resnet50"
    return model

def unet_efficientnetb3(input_shape=(512, 512, 3), num_classes=8, learning_rate=1e-4):
    """
    Modèle U-Net avec backbone EfficientNet-B3 pré-entraîné sur ImageNet.
    """
    sm.set_framework("tf.keras")

    model = sm.Unet(
        backbone_name="efficientnetb3",
        encoder_weights="imagenet",
        input_shape=input_shape,
        classes=num_classes,
        activation="softmax"
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=sm.losses.categorical_focal_jaccard_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.FScore(), dice_coef]
    )

    model._name = "unet_efficientnetb3"
    return model

def charger_metriques(dossier_logs):
    """
    Charge tous les fichiers CSV de métriques présents dans un dossier.

    Args:
        dossier_logs (str): Chemin vers le dossier contenant les fichiers CSV.

    Returns:
        dict: Dictionnaire avec nom du modèle en clé et dataframe en valeur.
    """
    fichiers = glob.glob(os.path.join(dossier_logs, "*.csv"))
    resultats = {}

    for fichier in fichiers:
        # Récupère le nom complet du modèle (par exemple unet_mini, unet_vgg16)
        nom_modele = "_".join(os.path.basename(fichier).split("_")[:-2])
        df = pd.read_csv(fichier)
        resultats[nom_modele] = df

    return resultats

def tracer_metriques(resultats):
    """
    Trace les métriques des différents modèles sur des graphiques.

    Args:
        resultats (dict): Dictionnaire avec nom modèle et dataframe.
    """

    # Palette de couleurs spécifique pour chaque modèle
    couleurs = {
        "mini": "blue",
        "vgg16": "green",
        "resnet50": "red",
        "efficientnetb3": "purple"
    }

    plt.figure(figsize=(18, 18))

    # Graphique de Loss (Perte)
    plt.subplot(3, 2, 1)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        plt.plot(df["loss"], label=f"{modele} Train Loss", color=couleur, linestyle="--")
        plt.plot(df["val_loss"], label=f"{modele} Val Loss", color=couleur, linestyle="-")
    plt.title("Comparaison des Loss (Perte)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Graphique Mean IoU
    plt.subplot(3, 2, 2)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "mean_iou" in df.columns:
            plt.plot(df["mean_iou"], label=f"{modele} Train Mean IoU", color=couleur, linestyle="--")
            plt.plot(df["val_mean_iou"], label=f"{modele} Val Mean IoU", color=couleur, linestyle="-")
        elif "iou_score" in df.columns:
            plt.plot(df["iou_score"], label=f"{modele} Train IoU Score", color=couleur, linestyle="--")
            plt.plot(df["val_iou_score"], label=f"{modele} Val IoU Score", color=couleur, linestyle="-")
    plt.title("Comparaison du Mean IoU / IoU Score")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.legend()

    # Graphique Dice Coefficient
    plt.subplot(3, 2, 3)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "dice_coef" in df.columns:
            plt.plot(df["dice_coef"], label=f"{modele} Train Dice", color=couleur, linestyle="--")
            plt.plot(df["val_dice_coef"], label=f"{modele} Val Dice", color=couleur, linestyle="-")
    plt.title("Comparaison du Dice Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.grid(True)
    plt.legend()

    # Graphique Accuracy
    plt.subplot(3, 2, 4)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "accuracy" in df.columns:
            plt.plot(df["accuracy"], label=f"{modele} Train Accuracy", color=couleur, linestyle="--")
            plt.plot(df["val_accuracy"], label=f"{modele} Val Accuracy", color=couleur, linestyle="-")
    plt.title("Comparaison de l'Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Graphique Temps d'entraînement par modèle
    plt.subplot(3, 1, 3)
    temps_entrainement = {}
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "temps_total_sec" in df.columns:
            temps = df["temps_total_sec"].iloc[-1] / 60  # converti en minutes
            temps_entrainement[modele] = temps
            plt.bar(modele, temps, color=couleur)
            plt.text(modele, temps, f"{temps:.2f} min", ha="center", va="bottom")

    plt.title("Comparaison du Temps total d'entraînement (en minutes)")
    plt.ylabel("Temps (minutes)")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()