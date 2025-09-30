# pip install scikit-fuzzy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import skfuzzy as fuzz

# ========= PARAMETER =========
FILE_PATH = "data_premier.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10              # range K untuk evaluasi jumlah cluster (FCM)
N_RUNS = 20                       # berapa kali ulang tiap K (ambil FPC terbaik)

# Normalisasi / Standarisasi
# Pilihan: "minmax", "zscore", "l2", "zscore_l2", "robust", "robust_l2"
NORMALIZATION_MODE = "zscore_l2"

# PCA control (paksa 2D untuk visual)
USE_PCA = True
PCA_N_COMPONENTS = 2  # pastikan 2 dimensi

# Hyperparameter FCM
M_FUZZ = 2.0           # tingkat “fuzziness” (m=2 umum dipakai)
MAX_ITER = 1000
TOL = 1e-5

# ========= BACA DATA (versi RAW → agregasi per tim) =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS, sheet_name="Match")

teams_col = df.columns[0]
# >>> JANGAN HAPUS KODE KOMEN DI BAWAH <<<
# df_grouped = df.groupby(df.columns[0], as_index=False)[df.columns[1:8]].mean()
df_grouped = df.groupby(teams_col, as_index=False)[df.columns[1:5]].mean()

teams = df_grouped.iloc[:, 0].astype(str)
X = df_grouped.iloc[:, 1:].to_numpy()
feature_names = df_grouped.columns[1:].tolist()

# ========= NORMALISASI / STANDARISASI =========
mean = std = mins = maxs = med = iqr = None

if NORMALIZATION_MODE == "minmax":
    mins = X.min(axis=0); maxs = X.max(axis=0)
    den = maxs - mins; den[den == 0] = 1.0
    Xn = (X - mins) / den

elif NORMALIZATION_MODE == "zscore":
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    Xn = (X - mean) / std

elif NORMALIZATION_MODE == "l2":
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Xn = X / norms

elif NORMALIZATION_MODE == "zscore_l2":
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    Xz = (X - mean) / std
    norms = np.linalg.norm(Xz, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Xn = Xz / norms

elif NORMALIZATION_MODE == "robust":
    med = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1; iqr[iqr == 0] = 1.0
    Xn = (X - med) / iqr

elif NORMALIZATION_MODE == "robust_l2":
    med = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1; iqr[iqr == 0] = 1.0
    Xr = (X - med) / iqr
    norms = np.linalg.norm(Xr, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Xn = Xr / norms

else:
    raise ValueError("NORMALIZATION_MODE harus salah satu dari: 'minmax','zscore','l2','zscore_l2','robust','robust_l2'")

print("[INFO] Shape X (original):", X.shape)

# ========= PCA (paksa 2 komponen untuk visual) =========
if USE_PCA:
    pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver='full')
    X_work = pca.fit_transform(Xn)
    evr = pca.explained_variance_ratio_
    print("\n=== PCA Info ===")
    print("Explained variance ratio per komponen:", [f"{v:.4f}" for v in evr])
    print("Explained variance kumulatif          :", f"{np.sum(evr):.4f}")
else:
    X_work = Xn

print("[INFO] Shape X (for clustering):", X_work.shape)

# ========= TENTUKAN K OPTIMAL (berdasar FPC tertinggi) =========
# Catatan: skfuzzy.cmeans butuh data shape = (fitur, sampel)
X_fcm = X_work.T

rows = []
best_overall = {
    "K": None, "FPC": -np.inf,
    "u": None, "centroids": None,
    "labels_hard": None, "sil_avg": None, "sil_samples": None
}

print("\n[SEARCH-K] Evaluasi K menggunakan FPC:")
for k in range(K_MIN, K_MAX + 1):
    best_k = {"FPC": -np.inf}

    # Multiple restarts → ambil FPC terbaik
    for r in range(N_RUNS):
        # init=None → random; set seed agar reproducible across runs
        seed = np.random.randint(0, 10_000_000)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X_fcm,
            c=k,
            m=M_FUZZ,
            error=TOL,
            maxiter=MAX_ITER,
            init=None,
            seed=seed
        )

        if fpc > best_k.get("FPC", -np.inf):
            labels_hard = np.argmax(u, axis=0)  # hard assignment untuk evaluasi/printing
            # Silhouette butuh ≥2 cluster dan setiap cluster punya anggota
            sil_avg = None; sil_samples = None
            try:
                if k >= 2 and len(np.unique(labels_hard)) == k:
                    sil_samples = silhouette_samples(X_work, labels_hard)
                    sil_avg = silhouette_score(X_work, labels_hard)
            except Exception:
                # Silhouette bisa gagal pada degenerate case; aman abaikan
                sil_avg = None; sil_samples = None

            best_k.update({
                "FPC": fpc,
                "u": u,
                "centroids": cntr,             # di ruang kerja (PCA jika USE_PCA=True)
                "labels_hard": labels_hard,
                "sil_avg": sil_avg,
                "sil_samples": sil_samples
            })

    rows.append({
        "K": k,
        "FPC": round(best_k["FPC"], 6),
        "Silhouette": None if best_k["sil_avg"] is None else round(float(best_k["sil_avg"]), 4)
    })

    # update best overall berdasarkan FPC (tie-breaker: silhouette lebih tinggi)
    choose = False
    if best_k["FPC"] > best_overall["FPC"]:
        choose = True
    elif np.isclose(best_k["FPC"], best_overall["FPC"], rtol=0, atol=1e-9):
        # tie-break: pilih yang silhouette rata-rata lebih tinggi jika tersedia
        sa_new = -np.inf if best_k["sil_avg"] is None else best_k["sil_avg"]
        sa_old = -np.inf if best_overall["sil_avg"] is None else best_overall["sil_avg"]
        choose = sa_new > sa_old

    if choose:
        best_overall.update({
            "K": k,
            "FPC": best_k["FPC"],
            "u": best_k["u"],
            "centroids": best_k["centroids"],
            "labels_hard": best_k["labels_hard"],
            "sil_avg": best_k["sil_avg"],
            "sil_samples": best_k["sil_samples"]
        })

df_k = pd.DataFrame(rows)
print("\n=== Ringkasan FPC per K (dan Silhouette hard-label bila ada) ===")
print(df_k.to_string(index=False))

K_optimal = best_overall["K"]
u = best_overall["u"]                       # membership matrix (shape: K x n_samples)
labels = best_overall["labels_hard"]        # hard labels via argmax
centroids_work = best_overall["centroids"]  # di ruang kerja (PCA jika aktif)
sil_avg = best_overall["sil_avg"]
sil_samples = best_overall["sil_samples"]

print(f"\n[SELECT] Memilih K={K_optimal} berdasarkan FPC tertinggi.")
print(f"FPC(K={K_optimal}) = {best_overall['FPC']:.6f}")
if sil_avg is not None:
    print(f"Silhouette (hard labels) = {sil_avg:.4f}")
else:
    print("Silhouette (hard labels) = N/A (degenerate atau tidak tersedia).")

# ========= Hasil per tim =========
df_result = pd.DataFrame({
    "Team": teams,
    "Cluster(hard)": labels + 1,
    "MaxMembership": np.max(u, axis=0).round(4)
})
if sil_samples is not None:
    df_result["Silhouette"] = np.round(sil_samples, 4)

print("\n=== Hasil Cluster per Tim (FCM hard-label) + Membership Max ===")
print(df_result.to_string(index=False))

# ========= Anggota cluster (pakai hard label) =========
print("\n=== Anggota Tiap Cluster (indeks 1-based; hard assignment) ===")
for cid in range(K_optimal):
    idxs = np.where(labels == cid)[0]
    idxs_1b = [int(i+1) for i in idxs]
    names = [teams.iloc[i] for i in idxs]
    print(f"Cluster {cid+1}: index={idxs_1b}")
    print("  Tim:", names)

# ========= CENTROID di ruang kerja & balik ke ruang normalisasi =========
print("\n=== CENTROID: RUANG KERJA ===")
if USE_PCA:
    for i, c in enumerate(centroids_work):
        print(f"Cluster {i+1} (PCA): {[round(float(v),4) for v in c]}")
else:
    for i, c in enumerate(centroids_work):
        pretty = ", ".join([f"{f}={v:.4f}" for f, v in zip(feature_names, c)])
        print(f"Cluster {i+1} (Std/Norm): {pretty}")

# balik ke ruang terstandardisasi (fitur asli)
if USE_PCA:
    centroids_norm = pca.inverse_transform(centroids_work)
else:
    centroids_norm = centroids_work.copy()

print("\n=== CENTROID: RUANG TERSTANDARISASI/NORMALISASI ===")
for i, c in enumerate(centroids_norm):
    pretty = ", ".join([f"{f}={v:.4f}" for f, v in zip(feature_names, c)])
    print(f"Cluster {i+1}: {pretty}")

# ========= PLOT 2D =========
print("\n[MPL] Scatter (FCM) — opacity merefleksikan kepastian (max membership).")

coords2d = X_work[:, :2] if X_work.shape[1] >= 2 else X_work
cents2d  = centroids_work[:, :2] if centroids_work.shape[1] >= 2 else centroids_work

plt.figure(figsize=(9,7))
palette = plt.cm.get_cmap('tab10', K_optimal)
colors = [palette(i) for i in range(K_optimal)]

# Opacity tiap titik = max membership (semakin “yakin”, semakin solid)
max_mem = np.max(u, axis=0)
for cid in range(K_optimal):
    idxs = np.where(labels == cid)[0]
    plt.scatter(coords2d[idxs, 0], coords2d[idxs, 1],
                s=70, alpha=np.clip(max_mem[idxs], 0.25, 1.0),
                color=colors[cid], label=f"Cluster {cid+1}")

    # centroid
    cx, cy = cents2d[cid]
    plt.scatter(cx, cy, s=280, marker='X', edgecolor='k', linewidths=1.2,
                color=colors[cid], zorder=5)

    # lingkaran radius = max jarak anggota (hard) ke centroid (visualisasi kasar)
    if len(idxs) > 0:
        dists = np.linalg.norm(coords2d[idxs] - [cx, cy], axis=1)
        radius = dists.max()
        circle = Circle((cx, cy), radius, color=colors[cid], fill=False, lw=2, alpha=0.5)
        plt.gca().add_patch(circle)

# anotasi tim (pakai sedikit offset)
for i in range(coords2d.shape[0]):
    plt.text(coords2d[i, 0], coords2d[i, 1], f" {teams.iloc[i]}", fontsize=8, va="center")

plt.title("Fuzzy C-Means (FCM) — warna=cluster (hard), opacity=max membership")
plt.xlabel("PC1" if USE_PCA else "Dim 1"); plt.ylabel("PC2" if USE_PCA else "Dim 2")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
