import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import numpy as np

# ====== (NEW) PCA ======
from sklearn.decomposition import PCA

# ========= PARAMETER =========
FILE_PATH = "data_premier.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10         # range K untuk elbow method
MAX_ITER = 200
TOL = 1e-6

# Normalisasi / Standarisasi
# Rekomendasi: "zscore" untuk PCA + KMeans
NORMALIZATION_MODE = "zscore"     # "minmax", "zscore", "l2", atau "zscore_l2"

# (NEW) PCA control
USE_PCA = True                    # True = aktifkan PCA, False = tidak
# PCA_N_COMPONENTS:
# - float di (0,1]: target proporsi varians (mis. 0.95 = 95% varians)
# - int >= 1: jumlah komponen
# - None: gunakan seluruh komponen (tidak direduksi)
PCA_N_COMPONENTS = 0.95

# MODE INISIALISASI CENTROID
INIT_MODE = "kmeans++"  # "kmeans++" atau "manual"
INIT_INDICES_1BASED = [1, 3, 5]  # centroid awal dari baris data ke-1,3,5 (1-based)

# ========= FUNGSI UTIL =========
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def compute_wcss(X, labels, centroids):
    return sum(euclidean(X[i], centroids[labels[i]])**2 for i in range(len(X)))

# --- helper untuk menampilkan centroid awal dalam TABEL skala asli
def show_init_centroids_table(indices, mode):
    """
    Tampilkan tabel centroid awal dalam skala ASLI (bukan PCA / standar).
    indices: list of 0-based row indices yang dipilih saat init
    mode: NORMALIZATION_MODE ('zscore' / 'minmax' / lainnya)
    Menggunakan variabel global: Xn, mean, std, mins, maxs, feature_names
    """
    rows = []
    for i in indices:
        if mode == "zscore":
            orig = Xn[i] * std + mean
        elif mode == "minmax":
            den = (maxs - mins).copy()
            den[den == 0] = 1.0
            orig = Xn[i] * den + mins
        else:
            # untuk 'l2' atau 'zscore_l2' tidak ada inverse unik -> tampilkan nilai terstandar
            orig = Xn[i]
        rows.append(orig)

    df_init = pd.DataFrame(rows, columns=feature_names, index=[f"Index {i+1}" for i in indices])
    print("\n[Centroid Awal di Skala Asli] (tabel):")
    print(df_init.round(4).to_string())

def init_centroids_kmeanspp(X, k):
    n = len(X)
    centroids = []
    chosen_idx = []
    # pilih centroid pertama (deterministik ke-0; bisa ganti random.randrange(n))
    first_idx = 0
    centroids.append(X[first_idx])
    chosen_idx.append(first_idx)

    for _ in range(1, k):
        dists_sq = []
        for x in X:
            dist_sq = min(euclidean(x, c) ** 2 for c in centroids)
            dists_sq.append(dist_sq)
        total = sum(dists_sq)
        if total == 0:
            # fallback kalau semua sama
            next_idx = random.randrange(n)
            centroids.append(X[next_idx])
            chosen_idx.append(next_idx)
            continue
        probs = [d / total for d in dists_sq]
        cumprobs = np.cumsum(probs)
        r = random.random()
        for i, p in enumerate(cumprobs):
            if r < p:
                centroids.append(X[i])
                chosen_idx.append(i)
                break

    # Cetak indeks & koordinat centroid awal (ruang kerja: PCA atau std)
    print("\n[INIT] Centroid awal (KMeans++) — indeks 1-based:", [i+1 for i in chosen_idx])
    for idx in chosen_idx:
        formatted = [f"{val:.4f}" for val in X[idx]]
        print(f"  • Index {idx+1} -> {formatted}")

    # Tambahan: tampilkan centroid awal dalam TABEL skala asli
    show_init_centroids_table(chosen_idx, NORMALIZATION_MODE)

    return centroids, chosen_idx

def init_centroids_manual(X, indices_0based):
    n = len(X)
    for i in indices_0based:
        if i < 0 or i >= n:
            raise ValueError(f"Indeks manual di luar rentang data: {i+1} (1-based).")
    centroids = [X[i] for i in indices_0based]

    print("\n[INIT] Centroid awal (MANUAL) — indeks 1-based:", [i+1 for i in indices_0based])
    for i in indices_0based:
        formatted = [f"{val:.4f}" for val in X[i]]
        print(f"  • Index {i+1} -> {formatted}")

    # Tambahan: tampilkan centroid awal dalam TABEL skala asli
    show_init_centroids_table(indices_0based, NORMALIZATION_MODE)

    return centroids, indices_0based

# ========= KMEANS UTAMA =========
def kmeans_manual_iter(X, teams, k, feature_names=None, max_iter=MAX_ITER, tol=TOL, verbose=True,
                       init_mode="kmeans++", init_indices_0based=None):
    n = len(X)

    if init_mode == "manual":
        if init_indices_0based is None:
            raise ValueError("init_indices_0based wajib diisi untuk init_mode='manual'")
        if len(init_indices_0based) != k:
            raise ValueError(f"Jumlah init indices ({len(init_indices_0based)}) != k ({k})")
        centroids, init_indices = init_centroids_manual(X, init_indices_0based)
    else:
        centroids, init_indices = init_centroids_kmeanspp(X, k)

    labels_old = [-1] * n

    for it in range(1, max_iter + 1):
        labels = []
        clusters = [[] for _ in range(k)]
        all_dists = []

        for x in X:
            dists = [euclidean(x, c) for c in centroids]
            cid = int(np.argmin(dists))
            labels.append(cid)
            clusters[cid].append(x)
            all_dists.append(dists)

        new_centroids = []
        for c in clusters:
            if c:
                dim = len(c[0])
                new_centroids.append([sum(row[j] for row in c)/len(c) for j in range(dim)])
            else:
                new_centroids.append(list(X[random.randrange(n)]))

        wcss = compute_wcss(X, labels, new_centroids)

        if verbose:
            print(f"\n=== Iterasi {it} ===")
            for i, dists in enumerate(all_dists):
                team_name = teams.iloc[i] if teams is not None else f"Data {i+1}"
                dist_str = ", ".join([f"{dist:.4f}" for dist in dists])
                print(f"{team_name}: [{dist_str}] -> Cluster {labels[i]+1}")

            print("\nCentroids (ruang kerja saat ini):")
            for idx, c in enumerate(new_centroids):
                rounded = [round(float(v), 4) for v in c]
                print(f"Centroid {idx+1}: {rounded}")

        # konvergensi sederhana
        if labels == labels_old:
            if verbose:
                print("\nCluster tidak berubah lagi. Iterasi selesai.")
            break

        labels_old = labels
        centroids = new_centroids

    # Kembalikan juga struktur klaster (anggota)
    members = [[] for _ in range(k)]
    for i, lab in enumerate(labels):
        members[lab].append(i)  # index 0-based

    return labels, centroids, wcss, init_indices, members

# ========= KMEANS UNTUK ELBOW =========
def kmeans_wcss(X, k, max_iter=10):
    _, _, wcss, _, _ = kmeans_manual_iter(X, None, k, max_iter=max_iter, verbose=False)
    return wcss

# ========= SILHOUETTE SCORES =========
def silhouette_scores(X, labels):
    n = len(X)
    scores = []
    for i in range(n):
        same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if same_cluster:
            a = sum(euclidean(X[i], X[j]) for j in same_cluster) / len(same_cluster)
        else:
            a = 0

        b = float("inf")
        for c in set(labels):
            if c == labels[i]:
                continue
            other_cluster = [j for j in range(n) if labels[j] == c]
            if other_cluster:
                dist = sum(euclidean(X[i], X[j]) for j in other_cluster) / len(other_cluster)
                if dist < b:
                    b = dist

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(s)
    return scores

# ========= BACA DATA (versi sudah dirata-rata) =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS, sheet_name="Mean", nrows=20)
teams = df.iloc[:, 0].astype(str)
num = df.iloc[:, 1:8].apply(pd.to_numeric, errors='coerce')
mask = num.notnull().all(axis=1)
teams = teams[mask].reset_index(drop=True)
X = num[mask].to_numpy()

# Simpan nama fitur untuk print rapi
feature_names = df.columns[1:1+X.shape[1]].tolist()

# ========= NORMALISASI / STANDARISASI =========
if NORMALIZATION_MODE == "minmax":
    mins = X.min(axis=0); maxs = X.max(axis=0)
    den = maxs - mins; den[den == 0] = 1.0
    Xn = (X - mins) / den

elif NORMALIZATION_MODE == "zscore":
    mean = X.mean(axis=0); std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

elif NORMALIZATION_MODE == "l2":
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

elif NORMALIZATION_MODE == "zscore_l2":  # Z-score per kolom → L2 per baris
    mean = X.mean(axis=0); std = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std
    norms = np.linalg.norm(Xz, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xz / norms

else:
    raise ValueError("NORMALIZATION_MODE harus 'minmax', 'zscore', 'l2', atau 'zscore_l2'")

# ========= (NEW) PCA =========
if USE_PCA:
    pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver='full')
    X_pca = pca.fit_transform(Xn)
    X_for_clustering = X_pca

    # Info PCA
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print("\n=== PCA Info ===")
    print(f"n_components terpakai : {getattr(pca, 'n_components_', pca.n_components)}")
    print("Explained variance ratio per komponen:", [f"{v:.4f}" for v in evr])
    print("Explained variance kumulatif          :", [f"{v:.4f}" for v in cum])

    # Plot explained variance (opsional)
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(evr)+1), cum, marker='o')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Komponen ke-")
    plt.ylabel("Kumulatif Varians")
    plt.grid(True, ls="--", alpha=0.4)
    plt.show()
else:
    X_for_clustering = Xn

print("\n[INFO] Shape X (normalized):", Xn.shape)
if USE_PCA:
    print("[INFO] Shape X (after PCA):", X_pca.shape)

# ========= ELBOW METHOD (di ruang PCA jika aktif) =========
wcss_list = []
diff_list = []
for k in range(K_MIN, K_MAX + 1):
    wcss = kmeans_wcss(X_for_clustering, k)
    wcss_list.append(wcss)
    diff_list.append(0 if len(wcss_list) == 1 else (wcss_list[-2] - wcss_list[-1]))

df_wcss = pd.DataFrame({
    "K": list(range(K_MIN, K_MAX + 1)),
    "WCSS": [round(w, 4) for w in wcss_list],
    "Diff": [round(d, 4) for d in diff_list]
})
print("\n=== Hasil WCSS & Selisih ===")
print(df_wcss.to_string(index=False))

max_diff_idx = np.argmax(diff_list[1:]) + 1
K_optimal = int(df_wcss.iloc[max_diff_idx]["K"])
print(f"\nK optimal (berdasarkan selisih WCSS terbesar) = {K_optimal}")

plt.figure(figsize=(8,5))
plt.plot(range(K_MIN, K_MAX+1), wcss_list, marker='o')
plt.title(f"Elbow Method ({NORMALIZATION_MODE.upper()} + {'PCA' if USE_PCA else 'No PCA'})")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ========= JALANKAN K-MEANS FINAL =========
if INIT_MODE == "manual":
    init_indices_0based = [i - 1 for i in INIT_INDICES_1BASED]
    K_manual = len(init_indices_0based)
    labels, centroids, wcss, init_indices, members = kmeans_manual_iter(
        X_for_clustering, teams, K_manual,
        verbose=True, init_mode="manual",
        init_indices_0based=init_indices_0based
    )
else:
    labels, centroids, wcss, init_indices, members = kmeans_manual_iter(
        X_for_clustering, teams, K_optimal,
        verbose=True, init_mode="kmeans++"
    )

# Tampilkan indeks centroid awal (1-based) lagi sebagai ringkasan
print("\n[SUMMARY] Indeks centroid awal (1-based):", [i+1 for i in init_indices])

# ========= SILHOUETTE (di ruang PCA jika aktif) =========
scores = silhouette_scores(X_for_clustering, labels)
df_result = pd.DataFrame({
    "Team": teams,
    "Cluster": [lbl+1 for lbl in labels],
    "Silhouette": [round(s, 4) for s in scores]
})
print("\n=== Hasil Cluster per Tim + Silhouette ===")
print(df_result.to_string(index=False))
print(f"\nSilhouette Score rata-rata = {sum(scores)/len(scores):.4f}")

# ========= CETAK ANGGOTA CLUSTER (indeks & nama) =========
print("\n=== Anggota Tiap Cluster (indeks 1-based) ===")
for cid, idxs in enumerate(members, start=1):
    idxs_1b = [i+1 for i in idxs]
    names = [teams.iloc[i] for i in idxs]
    print(f"Cluster {cid}: index={idxs_1b}")
    print("  Tim:", names)

# ========= CETAK CENTROID DI TIGA RUANG =========
# (1) RUANG KERJA
print("\n=== CENTROID: RUANG KERJA ===")
if USE_PCA:
    centroids_pca = np.array(centroids, dtype=float)
    for idx, c in enumerate(centroids_pca):
        print(f"Cluster {idx+1} (PCA): {[round(float(v), 4) for v in c]}")
else:
    centroids_norm = np.array(centroids, dtype=float)
    for idx, c in enumerate(centroids_norm):
        pretty = ", ".join([f"{fname}={val:.4f}" for fname, val in zip(feature_names, c)])
        print(f"Cluster {idx+1} (Std/Norm): {pretty}")

# (2) RUANG TERSTANDARISASI/NORMALISASI (dimensi = fitur asli)
if USE_PCA:
    centroids_in_norm_space = PCA.inverse_transform(pca, centroids_pca)  # sama seperti pca.inverse_transform
else:
    centroids_in_norm_space = np.array(centroids, dtype=float)

print("\n=== CENTROID: RUANG TERSTANDARISASI/NORMALISASI ===")
for idx, c in enumerate(centroids_in_norm_space):
    pretty = ", ".join([f"{fname}={val:.4f}" for fname, val in zip(feature_names, c)])
    print(f"Cluster {idx+1}: {pretty}")

# (3) RUANG ASLI (satuan fitur) bila memungkinkan
if NORMALIZATION_MODE == "zscore":
    centroids_orig = centroids_in_norm_space * std + mean
    print("\n=== CENTROID: RUANG ASLI (dari Z-score) ===")
    for idx, c in enumerate(centroids_orig):
        pretty = ", ".join([f"{fname}={val:.4f}" for fname, val in zip(feature_names, c)])
        print(f"Cluster {idx+1}: {pretty}")

elif NORMALIZATION_MODE == "minmax":
    den = (maxs - mins).copy()
    den[den == 0] = 1.0
    centroids_orig = centroids_in_norm_space * den + mins
    print("\n=== CENTROID: RUANG ASLI (dari Min-Max) ===")
    for idx, c in enumerate(centroids_orig):
        pretty = ", ".join([f"{fname}={val:.4f}" for fname, val in zip(feature_names, c)])
        print(f"Cluster {idx+1}: {pretty}")

# ========= SCATTER PLOT HASIL KMEANS =========
# Kita butuh koordinat 2D untuk plot. Strategi:
# - Jika ruang kerja (X_for_clustering) punya >=2 dimensi -> pakai dua dimensi pertama.
# - Jika tidak, buat PCA khusus visualisasi dari Xn (2 komponen) lalu proyeksikan juga centroid ke ruang tsb.

print("\n[MPL] Scatter plot hasil K-Means...")

def get_viz_coords_and_centroids():
    # koordinat titik (N x 2), centroid (K x 2), label sumbu, info sumber
    if USE_PCA and X_for_clustering.shape[1] >= 2:
        coords = X_for_clustering[:, :2]
        cents = np.array(centroids, dtype=float)[:, :2]
        axis_label = ("PC1", "PC2")
        source = "PCA (komponen 1-2)"
    elif (not USE_PCA) and X_for_clustering.shape[1] >= 2:
        coords = X_for_clustering[:, :2]
        cents = np.array(centroids, dtype=float)[:, :2]
        axis_label = ("Dim 1", "Dim 2")
        source = "Ruang terstandar (2 fitur awal)"
    else:
        # fallback: PCA khusus visualisasi dari Xn
        pca_vis = PCA(n_components=2, svd_solver='full')
        coords = pca_vis.fit_transform(Xn)
        # proyeksikan centroid ke ruang visualisasi:
        if USE_PCA:
            # centroids ada di PCA-space (d engan komponen USE_PCA). Kembalikan dulu ke ruang standar, lalu transform
            cents_norm = pca.inverse_transform(np.array(centroids, dtype=float))
        else:
            cents_norm = np.array(centroids, dtype=float)
        cents = pca_vis.transform(cents_norm)
        axis_label = ("Viz-PC1", "Viz-PC2")
        source = "PCA 2D khusus visualisasi"
    return coords, cents, axis_label, source

coords2d, cents2d, axis_label, source = get_viz_coords_and_centroids()

plt.figure(figsize=(8,6))
K = len(set(labels))
palette = plt.cm.get_cmap('tab10', K)

# plot titik
for cid in range(K):
    idxs = [i for i, lab in enumerate(labels) if lab == cid]
    plt.scatter(coords2d[idxs, 0], coords2d[idxs, 1], s=60, alpha=0.8, label=f"Cluster {cid+1}", color=palette(cid))

# anotasi nama tim
for i, name in enumerate(teams):
    plt.text(coords2d[i, 0], coords2d[i, 1], f" {name}", fontsize=8, va="center")

# plot centroid
plt.scatter(cents2d[:, 0], cents2d[:, 1], s=220, marker='X', edgecolor='k', linewidths=1.0, label="Centroid")

plt.title(f"K-Means Scatter Plot ({source})")
plt.xlabel(axis_label[0]); plt.ylabel(axis_label[1])
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
