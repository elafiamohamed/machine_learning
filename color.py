"""
=============================================================
  Affichage Vraie Couleur (True Color RGB) — Landsat 8 L2SP
  B4 (Rouge) + B3 (Vert) + B2 (Bleu)
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# ─────────────────────────────────────────────
# CONFIGURATION — adaptez les chemins si besoin
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "imageSat")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultats")
os.makedirs(OUTPUT_DIR, exist_ok=True)

B2_FILE = os.path.join(DATA_DIR, "LC08_L2SP_201035_20250702_20250711_02_T1_SR_B2.TIF")  # Bleu
B3_FILE = os.path.join(DATA_DIR, "LC08_L2SP_201035_20250702_20250711_02_T1_SR_B3.TIF")  # Vert
B4_FILE = os.path.join(DATA_DIR, "B4.TIF")  # Rouge

# Constantes L2SP (depuis le fichier MTL)
REFL_MULT = 2.75e-05
REFL_ADD  = -0.2


# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────

def load_band(filepath):
    with rasterio.open(filepath) as src:
        band = src.read(1).astype(np.float32)
    return band


def dn_to_reflectance(band_raw):
    """DN → Réflectance de surface L2SP, pixels no-data → NaN."""
    band = np.where(band_raw == 0, np.nan, band_raw)
    refl = REFL_MULT * band + REFL_ADD
    return np.clip(refl, 0.0, 1.0).astype(np.float32)


def percentile_stretch(band, pmin=2, pmax=98):
    """
    Étirement de contraste par percentiles pour améliorer la visualisation.
    Indispensable pour Landsat dont les réflectances sont faibles (~0.05–0.3).
    """
    lo = np.nanpercentile(band, pmin)
    hi = np.nanpercentile(band, pmax)
    stretched = (band - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)


def build_rgb(r, g, b):
    """Empile les 3 bandes en tableau HxWx3 prêt pour imshow."""
    rgb = np.dstack([r, g, b])
    return rgb


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("→ Chargement des bandes B2, B3, B4...")
    b2_raw = load_band(B2_FILE)
    b3_raw = load_band(B3_FILE)
    b4_raw = load_band(B4_FILE)

    print("→ Conversion en réflectance de surface (L2SP)...")
    b2 = dn_to_reflectance(b2_raw)   # Bleu
    b3 = dn_to_reflectance(b3_raw)   # Vert
    b4 = dn_to_reflectance(b4_raw)   # Rouge

    print("→ Étirement de contraste (percentile 2–98%)...")
    r = percentile_stretch(b4)
    g = percentile_stretch(b3)
    b = percentile_stretch(b2)

    rgb = build_rgb(r, g, b)

    print("→ Affichage...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_title(
        "Image Vraie Couleur (True Color RGB)\n"
        "Landsat 8 L2SP — B4/B3/B2 — 2 juillet 2025",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.axis("off")
    plt.tight_layout()

    # Sauvegarde
    out_path = os.path.join(OUTPUT_DIR, "true_color.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✔ Image sauvegardée → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()