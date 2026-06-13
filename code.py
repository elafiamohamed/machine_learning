"""
=============================================================
  Détection des Îlots de Chaleur Urbains — Landsat 8 L2SP
  Pipeline ML : Prétraitement → Dataset → Train/Test → Résultats
=============================================================
Structure du projet :
    projet/
        imageSat/     ← SR_B4.TIF, SR_B5.TIF, ST_B10.TIF, MTL.txt
        code.py       ← ce fichier
        resultats/    ← généré automatiquement
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import rasterio
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "imageSat")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultats")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Noms exacts des fichiers Landsat 8 L2SP
B4_FILE  = os.path.join(DATA_DIR, "B4.TIF")
B5_FILE  = os.path.join(DATA_DIR, "B5.TIF")
B10_FILE = os.path.join(DATA_DIR, "B10.TIF")

# Constantes L2SP lues dans le fichier MTL
# GROUP = LEVEL2_SURFACE_TEMPERATURE_PARAMETERS
TEMP_MULT = 0.00341802   # TEMPERATURE_MULT_BAND_ST_B10
TEMP_ADD  = 149.0        # TEMPERATURE_ADD_BAND_ST_B10

# Constantes L2SP pour la réflectance de surface (B4, B5)
REFL_MULT = 2.75e-05     # REFLECTANCE_MULT_BAND_4 / 5
REFL_ADD  = -0.2         # REFLECTANCE_ADD_BAND_4 / 5

N_CLUSTERS  = 3          # Zones : Fraîche / Moyenne / Chaude
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════
# 1. PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════

def load_band(filepath):
    """Charge une bande raster et retourne (array 2D float32, profil rasterio)."""
    with rasterio.open(filepath) as src:
        band    = src.read(1).astype(np.float32)
        profile = src.profile
    return band, profile


def convert_b4b5_to_reflectance(band_raw):
    """
    Conversion DN → Réflectance de Surface (L2SP Collection 2).
    ρ = REFLECTANCE_MULT * DN + REFLECTANCE_ADD
    Pixels No-Data (DN=0) → NaN
    """
    band = np.where(band_raw == 0, np.nan, band_raw)
    refl = REFL_MULT * band + REFL_ADD
    # Clamp physique [0, 1]
    refl = np.clip(refl, 0.0, 1.0)
    return refl.astype(np.float32)


def convert_b10_to_lst(b10_raw):
    """
    Conversion directe L2SP : DN → LST (°C).
    Produit ST (Surface Temperature) — USGS a déjà corrigé l'atmosphère.
    LST(K) = TEMPERATURE_MULT * DN + TEMPERATURE_ADD
    LST(°C) = LST(K) - 273.15
    Pixels No-Data (DN=0) → NaN
    """
    b10 = np.where(b10_raw == 0, np.nan, b10_raw)
    lst_k = TEMP_MULT * b10 + TEMP_ADD
    lst_c = lst_k - 273.15
    return lst_c.astype(np.float32)


def compute_ndvi(b4_refl, b5_refl):
    """NDVI = (NIR - RED) / (NIR + RED) à partir des réflectances."""
    denom = b5_refl + b4_refl
    ndvi  = np.where(denom > 0, (b5_refl - b4_refl) / denom, np.nan)
    return ndvi.astype(np.float32)


def preprocess():
    """Exécute tout le prétraitement et retourne NDVI, LST, profil."""
    print("=" * 57)
    print("  ÉTAPE 1 — PRÉTRAITEMENT")
    print("=" * 57)

    print("  → Chargement des bandes B4, B5, B10...")
    b4_raw,  profile = load_band(B4_FILE)
    b5_raw,  _       = load_band(B5_FILE)
    b10_raw, _       = load_band(B10_FILE)

    print("  → Conversion B4 / B5 → Réflectance de surface...")
    b4 = convert_b4b5_to_reflectance(b4_raw)
    b5 = convert_b4b5_to_reflectance(b5_raw)

    print("  → Calcul NDVI...")
    ndvi = compute_ndvi(b4, b5)

    print("  → Conversion B10 (ST_B10 L2SP) → LST (°C)...")
    lst = convert_b10_to_lst(b10_raw)

    print(f"\n  ✔ Résolution image   : {lst.shape[0]} × {lst.shape[1]} pixels")
    print(f"  ✔ LST  → min={np.nanmin(lst):.1f}°C  max={np.nanmax(lst):.1f}°C  "
          f"moy={np.nanmean(lst):.1f}°C")
    print(f"  ✔ NDVI → min={np.nanmin(ndvi):.3f}  max={np.nanmax(ndvi):.3f}  "
          f"moy={np.nanmean(ndvi):.3f}\n")

    return ndvi, lst, profile


# ═══════════════════════════════════════════════════════════
# 2. EXTRACTION DU DATASET
# ═══════════════════════════════════════════════════════════

def build_dataset(ndvi, lst):
    """
    Construit la matrice de features X = [LST, NDVI] 
    à partir des pixels valides uniquement.
    """
    print("=" * 57)
    print("  ÉTAPE 2 — EXTRACTION DU DATASET")
    print("=" * 57)

    rows, cols = lst.shape
    lst_flat   = lst.ravel()
    ndvi_flat  = ndvi.ravel()

    # Masque pixels valides : pas de NaN, plage physique réaliste
    valid_mask = (
        np.isfinite(lst_flat)   &
        np.isfinite(ndvi_flat)  &
        (lst_flat > -20.0)      &   # seuil bas (gel extrême)
        (lst_flat <  80.0)      &   # seuil haut (aberrant)
        (ndvi_flat > -1.0)      &
        (ndvi_flat <  1.0)
    )

    X_valid   = np.column_stack([lst_flat[valid_mask],
                                  ndvi_flat[valid_mask]])
    idx_valid = np.where(valid_mask)[0]

    print(f"  ✔ Pixels totaux    : {rows * cols:,}")
    print(f"  ✔ Pixels valides   : {X_valid.shape[0]:,}  "
          f"({X_valid.shape[0] / (rows*cols) * 100:.1f}%)")
    print(f"  ✔ Features         : [LST (°C), NDVI]")
    print(f"  ✔ Shape dataset    : {X_valid.shape}\n")

    return X_valid, idx_valid, valid_mask, rows, cols


def normalize_dataset(X):
    """Normalisation StandardScaler : μ=0, σ=1 sur chaque feature."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ═══════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT & TEST
# ═══════════════════════════════════════════════════════════

def find_optimal_k(X_scaled, k_range=range(2, 8)):
    """
    Méthode du coude + Silhouette pour visualiser le meilleur K.
    (Optionnel — décommentez l'appel dans main() si besoin)
    """
    print("  → Recherche du K optimal (coude + silhouette)...")
    inertias, silhouettes = [], []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=10000))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="#E63946")
    axes[0].set_title("Méthode du Coude (Inertie)")
    axes[0].set_xlabel("Nombre de clusters K")
    axes[0].set_ylabel("Inertie")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "s-", color="#2A9D8F")
    axes[1].set_title("Score de Silhouette")
    axes[1].set_xlabel("Nombre de clusters K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "00_choix_K.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Graphe K optimal → {path}\n")


def train_and_evaluate(X_scaled):
    """
    Pipeline ML complet :
      1. Split 80% train / 20% test
      2. K-Means entraîné sur train uniquement
      3. Prédiction et métriques sur test
      4. Prédiction sur l'ensemble pour la carte finale
    """
    print("=" * 57)
    print("  ÉTAPE 3 — ENTRAÎNEMENT & ÉVALUATION")
    print("=" * 57)

    # ── Split ────────────────────────────────────────────
    X_train, X_test = train_test_split(
        X_scaled, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"  ✔ Train : {X_train.shape[0]:,} pixels  (80%)")
    print(f"  ✔ Test  : {X_test.shape[0]:,} pixels  (20%)\n")

    # ── Entraînement K-Means sur TRAIN ───────────────────
    print(f"  → Entraînement K-Means (K={N_CLUSTERS}, init=k-means++)...")
    kmeans = KMeans(
        n_clusters  = N_CLUSTERS,
        init        = "k-means++",
        n_init      = 15,
        max_iter    = 300,
        random_state= RANDOM_SEED
    )
    kmeans.fit(X_train)
    print(f"  ✔ Inertie (train)    : {kmeans.inertia_:.2f}")

    # ── Prédiction sur TEST ───────────────────────────────
    labels_test = kmeans.predict(X_test)

    # ── Métriques ─────────────────────────────────────────
    sil_train = silhouette_score(X_train, kmeans.labels_, sample_size=10000)
    sil_test  = silhouette_score(X_test,  labels_test,    sample_size=10000)
    db_train  = davies_bouldin_score(X_train, kmeans.labels_)
    db_test   = davies_bouldin_score(X_test,  labels_test)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  MÉTRIQUES D'ÉVALUATION                     │")
    print(f"  ├──────────────────────┬───────────┬──────────┤")
    print(f"  │ Métrique             │   Train   │   Test   │")
    print(f"  ├──────────────────────┼───────────┼──────────┤")
    print(f"  │ Silhouette Score  ↑  │  {sil_train:.4f}   │  {sil_test:.4f}  │")
    print(f"  │ Davies-Bouldin    ↓  │  {db_train:.4f}   │  {db_test:.4f}  │")
    print(f"  └──────────────────────┴───────────┴──────────┘\n")

    # ── Prédiction sur TOUT le dataset (pour la carte) ───
    labels_all = kmeans.predict(X_scaled)

    metrics = {
        "sil_train": sil_train, "sil_test": sil_test,
        "db_train" : db_train,  "db_test" : db_test,
        "inertia"  : kmeans.inertia_
    }
    return kmeans, labels_all, metrics


# ═══════════════════════════════════════════════════════════
# 4. RÉSULTATS FINAUX
# ═══════════════════════════════════════════════════════════

def assign_thermal_labels(kmeans, scaler):
    """
    Associe chaque cluster à une étiquette thermique
    en triant les centroïdes par LST moyenne croissante.
    0 = Fraîche  |  1 = Moyenne  |  2 = Chaude
    """
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    lst_centers  = centers_orig[:, 0]   # colonne 0 = LST
    order        = np.argsort(lst_centers)

    label_map   = {order[0]: 0, order[1]: 1, order[2]: 2}
    label_names = {0: "Fraîche", 1: "Moyenne", 2: "Chaude"}

    print("  Centroïdes des clusters (espace original) :")
    for i, idx in enumerate(order):
        print(f"    Cluster {idx} → {label_names[i]:7s} | "
              f"LST={lst_centers[idx]:.1f}°C  "
              f"NDVI={centers_orig[idx, 1]:.3f}")
    print()

    return label_map, label_names, lst_centers


def remap_labels(labels_all, label_map):
    """Remappage : clusters K-Means → étiquettes thermiques 0/1/2."""
    return np.vectorize(label_map.get)(labels_all)


def save_results(ndvi, lst, labels_all, label_map, label_names,
                 idx_valid, rows, cols, profile, metrics):
    """Génère toutes les visualisations et exports."""
    print("=" * 57)
    print("  ÉTAPE 4 — RÉSULTATS FINAUX")
    print("=" * 57)

    labels_remapped = remap_labels(labels_all, label_map)

    # ── Reconstruction carte pleine résolution ────────────
    full_map = np.full(rows * cols, np.nan)
    full_map[idx_valid] = labels_remapped
    full_map = full_map.reshape(rows, cols)

    # ── Palette thermique ────────────────────────────────
    cmap_th = mcolors.ListedColormap(["#2166AC", "#FEE08B", "#D73027"])
    norm_th = mcolors.BoundaryNorm([0, 1, 2, 3], cmap_th.N)

    # ─────────────────────────────────────────────────────
    # Figure 1 : LST
    # ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(lst, cmap="RdYlBu_r",
                   vmin=np.nanpercentile(lst, 2),
                   vmax=np.nanpercentile(lst, 98))
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Température (°C)", fontsize=11)
    ax.set_title("Land Surface Temperature (LST)", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_LST.png"), dpi=150)
    plt.close()
    print("  ✔ 01_LST.png")

    # ─────────────────────────────────────────────────────
    # Figure 2 : NDVI
    # ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("NDVI", fontsize=11)
    ax.set_title("NDVI — Indice de Végétation Normalisé",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_NDVI.png"), dpi=150)
    plt.close()
    print("  ✔ 02_NDVI.png")

    # ─────────────────────────────────────────────────────
    # Figure 3 : Carte principale des îlots de chaleur
    # ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(full_map, cmap=cmap_th, norm=norm_th,
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5],
                        fraction=0.035, pad=0.04)
    cbar.ax.set_yticklabels(
        [label_names[0], label_names[1], label_names[2]], fontsize=12)
    cbar.set_label("Zone Thermique", fontsize=12)
    ax.set_title("Détection des Îlots de Chaleur Urbains\n"
                 "Landsat 8 L2SP — K-Means (K=3)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    path_main = os.path.join(OUTPUT_DIR, "03_carte_ilots_chaleur.png")
    plt.savefig(path_main, dpi=200)
    plt.close()
    print(f"  ✔ 03_carte_ilots_chaleur.png  ← CARTE PRINCIPALE")

    # ─────────────────────────────────────────────────────
    # Figure 4 : Dashboard comparatif 3 couches
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    im0 = axes[0].imshow(lst, cmap="RdYlBu_r",
                          vmin=np.nanpercentile(lst, 2),
                          vmax=np.nanpercentile(lst, 98))
    plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.04, label="°C")
    axes[0].set_title("LST (°C)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.04, label="NDVI")
    axes[1].set_title("NDVI", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(full_map, cmap=cmap_th, norm=norm_th,
                          interpolation="nearest")
    cb2 = plt.colorbar(im2, ax=axes[2], ticks=[0.5, 1.5, 2.5],
                        fraction=0.035, pad=0.04)
    cb2.ax.set_yticklabels([label_names[0], label_names[1], label_names[2]])
    axes[2].set_title("Zones Thermiques", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Détection des Îlots de Chaleur Urbains — Landsat 8 L2SP",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ 04_dashboard.png")

    # ─────────────────────────────────────────────────────
    # Figure 5 : Statistiques par zone
    # ─────────────────────────────────────────────────────
    colors = ["#2166AC", "#FEE08B", "#D73027"]
    names  = [label_names[i] for i in range(3)]
    counts = [(labels_remapped == i).sum() for i in range(3)]
    pcts   = [c / len(labels_remapped) * 100 for c in counts]

    lst_flat = lst.ravel()
    data_per_zone = [
        lst_flat[idx_valid[labels_remapped == i]] for i in range(3)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Barres
    bars = axes[0].bar(names, pcts, color=colors, edgecolor="white",
                       linewidth=1.5, width=0.5)
    axes[0].set_ylabel("% de pixels valides", fontsize=11)
    axes[0].set_title("Répartition des Zones Thermiques", fontsize=12,
                       fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, max(pcts) * 1.18)
    for i, (p, c) in enumerate(zip(pcts, counts)):
        axes[0].text(i, p + 0.4, f"{p:.1f}%\n({c:,} px)",
                     ha="center", va="bottom", fontsize=9)

    # Boxplots
    bp = axes[1].boxplot(data_per_zone, labels=names, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[1].set_ylabel("LST (°C)", fontsize=11)
    axes[1].set_title("Distribution LST par Zone", fontsize=12,
                       fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_statistiques.png"), dpi=150)
    plt.close()
    print("  ✔ 05_statistiques.png")

    # ─────────────────────────────────────────────────────
    # Export GeoTIFF géoréférencé (ouvrable dans QGIS)
    # ─────────────────────────────────────────────────────
    profile_out = profile.copy()
    profile_out.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    geotiff_path = os.path.join(OUTPUT_DIR, "carte_ilots_chaleur.tif")
    with rasterio.open(geotiff_path, "w", **profile_out) as dst:
        dst.write(full_map.astype(np.float32), 1)
    print(f"  ✔ carte_ilots_chaleur.tif  (GeoTIFF — ouvrable QGIS)")

    # ─────────────────────────────────────────────────────
    # Rapport texte
    # ─────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "rapport.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  RAPPORT — DÉTECTION ÎLOTS DE CHALEUR URBAINS\n")
        f.write("  Landsat 8 L2SP — 2025-07-02 — WRS 201/035\n")
        f.write("=" * 55 + "\n\n")
        f.write("PARAMÈTRES\n")
        f.write(f"  Produit          : L2SP Collection 2 (ST_B10)\n")
        f.write(f"  TEMP_MULT        : {TEMP_MULT}\n")
        f.write(f"  TEMP_ADD         : {TEMP_ADD}\n")
        f.write(f"  N_CLUSTERS       : {N_CLUSTERS}\n\n")
        f.write("MÉTRIQUES ML\n")
        f.write(f"  Silhouette Train : {metrics['sil_train']:.4f}\n")
        f.write(f"  Silhouette Test  : {metrics['sil_test']:.4f}\n")
        f.write(f"  Davies-Bouldin Train : {metrics['db_train']:.4f}\n")
        f.write(f"  Davies-Bouldin Test  : {metrics['db_test']:.4f}\n")
        f.write(f"  Inertie          : {metrics['inertia']:.2f}\n\n")
        f.write("ZONES THERMIQUES\n")
        for i in range(3):
            lst_moy = np.nanmean(data_per_zone[i])
            lst_min = np.nanmin(data_per_zone[i])
            lst_max = np.nanmax(data_per_zone[i])
            f.write(f"  {label_names[i]:7s} : {pcts[i]:5.1f}%  "
                    f"({counts[i]:,} px)  "
                    f"LST moy={lst_moy:.1f}°C  "
                    f"[{lst_min:.1f} – {lst_max:.1f}°C]\n")
    print(f"  ✔ rapport.txt\n")

    # ── Résumé console ────────────────────────────────────
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  RÉSUMÉ ZONES THERMIQUES                         │")
    print("  ├─────────┬────────┬──────────────────────────────-┤")
    print("  │ Zone    │   %    │  LST moy    min     max        │")
    print("  ├─────────┼────────┼───────────────────────────────┤")
    for i in range(3):
        m  = np.nanmean(data_per_zone[i])
        mn = np.nanmin(data_per_zone[i])
        mx = np.nanmax(data_per_zone[i])
        print(f"  │ {label_names[i]:7s} │ {pcts[i]:5.1f}% │  "
              f"{m:5.1f}°C   {mn:5.1f}°C  {mx:5.1f}°C       │")
    print("  └─────────┴────────┴───────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════
# MAIN — Orchestration du pipeline
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 57)
    print("  DÉTECTION ÎLOTS DE CHALEUR URBAINS — LANDSAT 8 L2SP")
    print("  Pipeline ML Complet")
    print("═" * 57 + "\n")

    # ÉTAPE 1 : Prétraitement
    ndvi, lst, profile = preprocess()

    # ÉTAPE 2 : Extraction du dataset
    X_raw, idx_valid, valid_mask, rows, cols = build_dataset(ndvi, lst)
    X_scaled, scaler = normalize_dataset(X_raw)

    # (Optionnel) Décommentez pour visualiser le choix de K :
    # find_optimal_k(X_scaled)

    # ÉTAPE 3 : Entraînement & évaluation
    kmeans, labels_all, metrics = train_and_evaluate(X_scaled)

    # ÉTAPE 4 : Résultats
    label_map, label_names, lst_centers = assign_thermal_labels(kmeans, scaler)
    save_results(ndvi, lst, labels_all, label_map, label_names,
                 idx_valid, rows, cols, profile, metrics)

    print("  ✅  Pipeline terminé — résultats dans → resultats/\n")


if __name__ == "__main__":
    main()