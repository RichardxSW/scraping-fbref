import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.patches import Circle

# ========= PARAMETER =========
FILE_PATH = "data_premier.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10              # range K untuk evaluasi jumlah cluster

# Normalisasi / Standarisasi
# Pilihan: "minmax", "zscore", "l2", "zscore_l2", "robust", "robust_l2"
NORMALIZATION_MODE = "robust_l2"

# PCA control (paksa 2D untuk visual)
USE_PCA = True
PCA_N_COMPONENTS = 2  # pastikan 2 dimensi

# Berapa kali KMeans diulang untuk mencari solusi terbaik (silhouette tertinggi)
N_RUNS = 20

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
    print("Explained variance kumulatif          :", [f"{np.cumsum(evr)[-1]:.4f}"])
else:
    X_work = Xn

print("[INFO] Shape X (for clustering):", X_work.shape)

# ========= TENTUKAN K OPTIMAL (Silhouette terbesar) =========
rows = []
best_k_record = {"K": None, "sil_avg": -1.0, "labels": None, "centroids": None, "sil_samples": None}

print("\n[SEARCH-K] Evaluasi K menggunakan Silhouette:")
for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, init="k-means++", n_init="auto")
    labels_k = km.fit_predict(X_work)
    sil_samples_k = silhouette_samples(X_work, labels_k)
    sil_avg_k = silhouette_score(X_work, labels_k)
    rows.append({"K": k, "Silhouette": round(sil_avg_k, 4)})
    if sil_avg_k > best_k_record["sil_avg"]:
        best_k_record.update({
            "K": k,
            "sil_avg": sil_avg_k,
            "labels": labels_k,
            "centroids": km.cluster_centers_,
            "sil_samples": sil_samples_k
        })

df_k = pd.DataFrame(rows)
print("\n=== Ringkasan Silhouette per K ===")
print(df_k.to_string(index=False))

K_optimal = best_k_record["K"]
labels = best_k_record["labels"]
centroids_work = best_k_record["centroids"]
sil_samples = best_k_record["sil_samples"]
sil_avg = best_k_record["sil_avg"]

print(f"\nK optimal (Silhouette) = {K_optimal}, dengan Silhouette rata-rata = {sil_avg:.4f}")

# ========= Silhouette =========
df_result = pd.DataFrame({"Team": teams, "Cluster": labels + 1, "Silhouette": np.round(sil_samples, 4)})
print("\n=== Hasil Cluster per Tim + Silhouette (SELECTED RUN) ===")
print(df_result.to_string(index=False))
print(f"\nSilhouette Score rata-rata = {sil_avg:.4f}")

# ========= Anggota cluster =========
print("\n=== Anggota Tiap Cluster (indeks 1-based) ===")
for cid in range(K_optimal):
    idxs = np.where(labels == cid)[0]
    idxs_1b = [int(i+1) for i in idxs]
    names = [teams.iloc[i] for i in idxs]
    print(f"Cluster {cid+1}: index={idxs_1b}")
    print("  Tim:", names)

# ========= Centroid di 3 ruang =========
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

# ========= SCATTER PLOT dengan lingkaran =========
print("\n[MPL] Scatter plot hasil K-Means dengan lingkaran...")

coords2d = X_work[:, :2] if X_work.shape[1] >= 2 else X_work
cents2d  = centroids_work[:, :2] if centroids_work.shape[1] >= 2 else centroids_work

plt.figure(figsize=(8,6))
palette = plt.cm.get_cmap('tab10', K_optimal)
colors = [palette(i) for i in range(K_optimal)]

for cid in range(K_optimal):
    idxs = np.where(labels == cid)[0]
    plt.scatter(coords2d[idxs, 0], coords2d[idxs, 1], s=60, alpha=0.85,
                color=colors[cid], label=f"Cluster {cid+1}")

    # centroid
    cx, cy = cents2d[cid]
    plt.scatter(cx, cy, s=260, marker='X', edgecolor='k', linewidths=1.2,
                color=colors[cid], zorder=5)

    # lingkaran radius = max jarak anggota ke centroid
    dists = np.linalg.norm(coords2d[idxs] - [cx, cy], axis=1)
    radius = dists.max()
    circle = Circle((cx, cy), radius, color=colors[cid], fill=False, lw=2, alpha=0.5)
    plt.gca().add_patch(circle)

    # anotasi tim
    for i in idxs:
        plt.text(coords2d[i, 0], coords2d[i, 1], f" {teams.iloc[i]}", fontsize=8, va="center")

plt.title("Hasil Clustering K-Means")
plt.xlabel("PC1" if USE_PCA else "Dim 1"); plt.ylabel("PC2" if USE_PCA else "Dim 2")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
