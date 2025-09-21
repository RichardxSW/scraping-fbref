import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# ========= PARAMETER =========
FILE_PATH = "data_premier.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10              # range K untuk evaluasi jumlah cluster

# Normalisasi / Standarisasi
# Pilihan: "minmax", "zscore", "l2", "zscore_l2", "robust", "robust_l2"
# Rekomendasi: "zscore" untuk PCA + KMeans; "robust/robust_l2" jika banyak outlier
NORMALIZATION_MODE = "zscore_l2"

# PCA control (paksa 2D untuk visual)
USE_PCA = True
# USE_PCA = False
PCA_N_COMPONENTS = 2  # pastikan 2 dimensi

# Berapa kali KMeans diulang untuk mencari solusi terbaik (silhouette tertinggi)
N_RUNS = 20

# ========= BACA DATA (versi RAW → agregasi per tim) =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS, sheet_name="Match")

# ambil kolom: 0 = Team, sisanya = statistik (misalnya sampai kolom ke-7)
teams_col = df.columns[0]
# >>> JANGAN HAPUS KODE KOMEN DI BAWAH <<<
# df_grouped = df.groupby(df.columns[0], as_index=False)[df.columns[1:8]].mean()
df_grouped = df.groupby(teams_col, as_index=False)[df.columns[1:5]].mean()

teams = df_grouped.iloc[:, 0].astype(str)
X = df_grouped.iloc[:, 1:].to_numpy()
feature_names = df_grouped.columns[1:].tolist()

# ========= NORMALISASI / STANDARISASI =========
# simpan parameter untuk inverse bila memungkinkan
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

# ========= (REPLACED) PILIH K BERDASARKAN SILHOUETTE SCORE =========
# Menggantikan Elbow Method: coba K=K_MIN..K_MAX, ambil K dengan silhouette rata-rata terbesar
sil_k_list = []
neg_k_list = []
inertia_k_list = []
valid_K = []

n_samples = X_work.shape[0]
k_values = list(range(K_MIN, K_MAX + 1))

print("\n=== Evaluasi K berdasarkan Silhouette Score ===")
for k in k_values:
    # syarat minimal: k < n_samples dan k >= 2
    if k >= n_samples or k < 2:
        print(f"K={k}: dilewati (k harus < jumlah sampel={n_samples} dan >=2)")
        sil_k_list.append(float("-inf"))
        neg_k_list.append(np.inf)
        inertia_k_list.append(np.inf)
        continue

    km_tmp = KMeans(n_clusters=k, init="k-means++", n_init="auto")
    labels_tmp = km_tmp.fit_predict(X_work)

    # Pastikan ada >1 label unik (teoretis selalu >1 jika k>=2 dan data valid)
    if len(np.unique(labels_tmp)) < 2:
        print(f"K={k}: cluster kurang dari 2 label unik, dilewati")
        sil_k_list.append(float("-inf"))
        neg_k_list.append(np.inf)
        inertia_k_list.append(np.inf)
        continue

    sil_samp_tmp = silhouette_samples(X_work, labels_tmp)
    sil_avg_tmp = silhouette_score(X_work, labels_tmp)
    neg_cnt_tmp = int(np.sum(sil_samp_tmp < 0))
    inertia_tmp = float(km_tmp.inertia_)

    sil_k_list.append(sil_avg_tmp)
    neg_k_list.append(neg_cnt_tmp)
    inertia_k_list.append(inertia_tmp)
    valid_K.append(k)

    print(f"K={k}: sil_avg={sil_avg_tmp:.4f}, neg_count={neg_cnt_tmp}, inertia={inertia_tmp:.4f}")

# Tabel ringkas
df_k = pd.DataFrame({
    "K": k_values,
    "Silhouette_Avg": [None if np.isneginf(v) else round(v, 6) for v in sil_k_list],
    "Negatives": [None if (np.isneginf(sil_k_list[i]) or np.isinf(neg_k_list[i])) else int(neg_k_list[i]) for i in range(len(k_values))],
    "Inertia": [None if np.isneginf(sil_k_list[i]) else round(inertia_k_list[i], 4) for i in range(len(k_values))]
})
print("\n=== Ringkasan Silhouette per K ===")
print(df_k.to_string(index=False))

# Seleksi K terbaik:
# 1) Maksimalkan silhouette rata-rata
# 2) Jika tie: minimalisir neg_count
# 3) Jika tie lagi: minimalisir inertia
best_idx = None
best_tuple = None  # (sil, -neg (dibalik jadi lebih besar lebih baik), -inertia (dibalik), k)
for i, k in enumerate(k_values):
    sil = sil_k_list[i]
    neg = neg_k_list[i]
    inert = inertia_k_list[i]
    if np.isneginf(sil):
        continue
    # gunakan tuple pembanding: lebih besar lebih baik
    key = (sil, -neg, -inert)
    if (best_tuple is None) or (key > best_tuple):
        best_tuple = key
        best_idx = i

if best_idx is None:
    raise RuntimeError("Gagal menentukan K optimal berdasarkan silhouette. Periksa data/input Anda.")

K_optimal = int(k_values[best_idx])
print(f"\nK optimal (silhouette rata-rata terbesar) = {K_optimal}")

# Plot kurva silhouette vs K
plt.figure(figsize=(8,5))
plt.plot(k_values, [(-1 if np.isneginf(v) else v) for v in sil_k_list], marker='o')
plt.title(f"Silhouette vs K ({NORMALIZATION_MODE.upper()} + {'PCA' if USE_PCA else 'No PCA'})")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Rata-rata")
plt.grid(True, ls="--", alpha=0.4)
plt.show()

# ========= K-MEANS: MULTI-RUN & PILIH TERBAIK (dengan syarat no-negative) =========
best_any = {  # fallback jika tidak ada yang memenuhi no-negative
    "sil_avg": -1.0,
    "neg_count": np.inf,
    "inertia": np.inf,
    "labels": None,
    "centroids": None,
    "sil_samples": None
}
best_no_neg = None  # kandidat terbaik dengan semua silhouette >= 0

print(f"\n[SEARCH] Mencari hasil terbaik dari {N_RUNS} run KMeans (init='k-means++', n_init='auto'):")
for run in range(1, N_RUNS+1):
    km = KMeans(n_clusters=K_optimal, init="k-means++", n_init="auto")  # tanpa random_state
    labels_run = km.fit_predict(X_work)
    sil_samples_run = silhouette_samples(X_work, labels_run)
    sil_avg_run = silhouette_score(X_work, labels_run)
    neg_count_run = int(np.sum(sil_samples_run < 0))
    inertia_run = float(km.inertia_)

    print(f"  Run {run:02d}: sil_avg={sil_avg_run:.4f}, neg_count={neg_count_run}, inertia={inertia_run:.4f}")

    # update best_any (fallback) dengan kriteria lama
    better_any = False
    if sil_avg_run > best_any["sil_avg"]:
        better_any = True
    elif np.isclose(sil_avg_run, best_any["sil_avg"], atol=1e-6):
        if neg_count_run < best_any["neg_count"]:
            better_any = True
        elif (neg_count_run == best_any["neg_count"]) and (inertia_run < best_any["inertia"]):
            better_any = True
    if better_any:
        best_any.update({
            "sil_avg": sil_avg_run,
            "neg_count": neg_count_run,
            "inertia": inertia_run,
            "labels": labels_run,
            "centroids": km.cluster_centers_,
            "sil_samples": sil_samples_run
        })

    # kumpulkan kandidat yang tidak punya silhouette negatif
    if neg_count_run == 0:
        if (best_no_neg is None) or (sil_avg_run > best_no_neg["sil_avg"]) or \
           (np.isclose(sil_avg_run, best_no_neg["sil_avg"], atol=1e-6) and inertia_run < best_no_neg["inertia"]):
            best_no_neg = {
                "sil_avg": sil_avg_run,
                "neg_count": neg_count_run,
                "inertia": inertia_run,
                "labels": labels_run,
                "centroids": km.cluster_centers_,
                "sil_samples": sil_samples_run
            }

# Ambil hasil sesuai syarat:
if best_no_neg is not None:
    selected = best_no_neg
    print("\n[SELECT] Memilih run DENGAN semua silhouette >= 0 (memenuhi syarat), dengan silhouette rata-rata tertinggi.")
else:
    selected = best_any
    print("\n[WARNING] Tidak ada run dengan semua silhouette >= 0 dalam N_RUNS.")
    print("          Dipilih fallback: silhouette rata-rata tertinggi, negatif paling sedikit, inertia terkecil.")

labels = selected["labels"]
centroids_work = selected["centroids"]
wcss_final = selected["inertia"]
sil_samples = selected["sil_samples"]
sil_avg = selected["sil_avg"]

print(f"\n[SUMMARY] Hasil terpilih → Silhouette rata-rata = {sil_avg:.4f}, "
      f"negatives = {int(np.sum(sil_samples < 0))}, inertia = {wcss_final:.4f}")

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

# coba ke ruang asli (jika bisa)
if NORMALIZATION_MODE == "zscore":
    centroids_orig = centroids_norm * std + mean
elif NORMALIZATION_MODE == "minmax":
    den = (maxs - mins).copy(); den[den == 0] = 1.0
    centroids_orig = centroids_norm * den + mins
elif NORMALIZATION_MODE == "robust":
    den = iqr.copy(); den[den == 0] = 1.0
    centroids_orig = centroids_norm * den + med
elif NORMALIZATION_MODE in ["l2", "zscore_l2", "robust_l2"]:
    centroids_orig = None  # tidak bisa inverse unik karena normalisasi per baris
else:
    centroids_orig = None

if centroids_orig is None:
    print("\n[INFO] Mode normalisasi ini tidak dapat di-inverse ke skala asli secara unik (karena normalisasi per baris).")
else:
    print("\n=== CENTROID: RUANG ASLI (satuan fitur) ===")
    for i, c in enumerate(centroids_orig):
        pretty = ", ".join([f"{f}={v:.4f}" for f, v in zip(feature_names, c)])
        print(f"Cluster {i+1}: {pretty}")

# ========= SCATTER PLOT (2D) =========
print("\n[MPL] Scatter plot hasil K-Means...")

coords2d = X_work[:, :2] if X_work.shape[1] >= 2 else X_work
cents2d  = centroids_work[:, :2] if centroids_work.shape[1] >= 2 else centroids_work

plt.figure(figsize=(8,6))
K = K_optimal
palette = plt.cm.get_cmap('tab10', K)
colors = [palette(i) for i in range(K)]

# titik
for cid in range(K):
    idxs = np.where(labels == cid)[0]
    plt.scatter(coords2d[idxs, 0], coords2d[idxs, 1], s=60, alpha=0.85,
                color=colors[cid], label=f"Cluster {cid+1}")

# anotasi nama tim
for i, name in enumerate(teams):
    plt.text(coords2d[i, 0], coords2d[i, 1], f" {name}", fontsize=8, va="center")

# centroid (warna sama)
for cid in range(K):
    plt.scatter(cents2d[cid, 0], cents2d[cid, 1], s=260, marker='X',
                edgecolor='k', linewidths=1.2, color=colors[cid],
                label=f"Centroid {cid+1}", zorder=5)

plt.title("Hasil Clustering K-Means")
plt.xlabel("PC1" if USE_PCA else "Dim 1"); plt.ylabel("PC2" if USE_PCA else "Dim 2")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
