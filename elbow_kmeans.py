import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import numpy as np

# ========= PARAMETER =========
FILE_PATH = "data_premier.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10         # range K untuk elbow method
MAX_ITER = 200
TOL = 1e-6
NORMALIZATION_MODE = "l2"     # "minmax", "zscore", atau "l2"
# NORMALIZATION_MODE = "l2"     # "minmax", "zscore", atau "l2"

# [NEW] MODE INISIALISASI CENTROID
INIT_MODE = "manual"  # "kmeans++" atau "manual"
INIT_INDICES_1BASED = [1, 3, 5]  # centroid awal dari baris data ke-1,3,5 (1-based)

# ========= FUNGSI UTIL =========
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def compute_wcss(X, labels, centroids):
    return sum(euclidean(X[i], centroids[labels[i]])**2 for i in range(len(X)))

def init_centroids_kmeanspp(X, k):
    n = len(X)
    centroids = []
    chosen_idx = []
    # pilih centroid pertama random
    # first_idx = random.randrange(n)
    first_idx=0
    centroids.append(X[first_idx])
    chosen_idx.append(first_idx)

    for _ in range(1, k):
        # hitung jarak minimum tiap titik ke centroid terdekat
        dists = []
        for x in X:
            dist_sq = min(euclidean(x, c) ** 2 for c in centroids)
            dists.append(dist_sq)

        # probabilitas proporsional terhadap kuadrat jarak
        probs = [d / sum(dists) for d in dists]
        cumprobs = np.cumsum(probs)

        r = random.random()
        for i, p in enumerate(cumprobs):
            if r < p:
                centroids.append(X[i])
                chosen_idx.append(i)
                break

    print("\nCentroid awal hasil KMeans++:")
    for idx in chosen_idx:
        formatted = [f"{val:.4f}" for val in X[idx]]
        print(f"Index {idx+1} -> {formatted}")

    return centroids, chosen_idx

# [NEW] Inisialisasi centroid manual
def init_centroids_manual(X, indices_0based):
    n = len(X)
    for i in indices_0based:
        if i < 0 or i >= n:
            raise ValueError(f"Indeks manual di luar rentang data: {i+1} (1-based).")
    centroids = [X[i] for i in indices_0based]
    print("\nCentroid awal (MANUAL):")
    for i in indices_0based:
        formatted = [f"{val:.4f}" for val in X[i]]
        print(f"Index {i+1} -> {formatted}")
    return centroids, indices_0based

# ========= KMEANS UTAMA =========
def kmeans_manual_iter(X, teams, k, feature_names=None, max_iter=MAX_ITER, tol=TOL, verbose=True
                       , init_mode="kmeans++", init_indices_0based=None  # [NEW]
                       ):
    n = len(X)
    # init_indices = list(range(k))
    # init_indices = random.sample(range(n), k)
    # init_indices=[0, 15, 18]
    # centroids = [X[i] for i in init_indices]

    # [NEW] pilih mode inisialisasi
    if init_mode == "manual":
        if init_indices_0based is None:
            raise ValueError("init_indices_0based wajib diisi untuk init_mode='manual'")
        if len(init_indices_0based) != k:
            raise ValueError(f"Jumlah init indices ({len(init_indices_0based)}) != k ({k})")
        centroids, init_indices = init_centroids_manual(X, init_indices_0based)
    else:
        centroids, init_indices = init_centroids_kmeanspp(X, k)

    labels_old = [-1] * n

    # === Print centroid awal ===
    # print("\n=== Centroid Awal ===")
    # for idx, c in enumerate(centroids):
    #     rounded = [round(float(v), 4) for v in c]
    #     if init_indices is not None:   # kalau manual
    #         print(f"Centroid {idx+1} (dari data index {init_indices[idx]}): {rounded}")
    #     else:  # kalau kmeans++
    #         print(f"Centroid {idx+1}: {rounded}")

    for it in range(1, max_iter + 1):
        labels = []
        clusters = [[] for _ in range(k)]
        all_dists = []

        for x in X:
            dists = [euclidean(x, c) for c in centroids]
            cid = dists.index(min(dists))
            labels.append(cid)
            clusters[cid].append(x)
            all_dists.append(dists)

        # update centroid
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
            # df_iter = pd.DataFrame(X, columns=[f"Feature{i+1}" for i in range(X.shape[1])])
            for i, dists in enumerate(all_dists):
                team_name = teams.iloc[i] if teams is not None else f"Data {i+1}"
                dist_str = ", ".join([f"{dist:.4f}" for dist in dists])
                print(f"{team_name}: [{dist_str}] -> Cluster {labels[i]+1}")

            print("\nCentroids:")
            for idx, c in enumerate(new_centroids):
                rounded = [round(float(v), 4) for v in c]
                print(f"Centroid {idx+1}: {rounded}")

        if labels == labels_old:
            if verbose:
                print("\nCluster tidak berubah lagi. Iterasi selesai.")
            break

        labels_old = labels
        centroids = new_centroids

    return labels, centroids, wcss

# ========= KMEANS UNTUK ELBOW =========
def kmeans_wcss(X, k, max_iter=10):
    _, _, wcss = kmeans_manual_iter(X, None, k, max_iter=max_iter, verbose=False)
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

# ========= BACA & NORMALISASI DATA (versi udh dirata rata)=========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS, sheet_name="Mean", nrows=10)
teams = df.iloc[:, 0].astype(str)
num = df.iloc[:, 1:7].apply(pd.to_numeric, errors='coerce')
mask = num.notnull().all(axis=1)
teams = teams[mask].reset_index(drop=True)
X = num[mask].to_numpy()

# ========= BACA & NORMALISASI DATA (versi raw) =========
# df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS, sheet_name="Match", nrows=10)
#
# # ambil kolom: 0 = Team, sisanya = statistik (misalnya 8 kolom)
# teams = df.iloc[:, 0].astype(str)
# num = df.iloc[:, 1:9].apply(pd.to_numeric, errors='coerce')
#
# # hanya ambil baris valid (tanpa NaN)
# mask = num.notnull().all(axis=1)
# df = df[mask].reset_index(drop=True)
#
# # groupby berdasarkan Team lalu rata-rata
# # df_grouped = df.groupby(df.iloc[:,0]).mean().reset_index()
# df_grouped = df.groupby(df.columns[0], as_index=False)[df.columns[1:9]].mean()
#
# # ambil nama tim
# teams = df_grouped.iloc[:, 0].astype(str)
#
# # ambil fitur numerik
# X = df_grouped.iloc[:, 1:].to_numpy()

if NORMALIZATION_MODE == "minmax":
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    den = maxs - mins
    den[den == 0] = 1.0
    Xn = (X - mins) / den

elif NORMALIZATION_MODE == "zscore":
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

elif NORMALIZATION_MODE == "l2":
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

elif NORMALIZATION_MODE == "zscore_l2":  # Z-score per kolom → L2 per baris
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std
    norms = np.linalg.norm(Xz, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xz / norms

else:
    raise ValueError("NORMALIZATION_MODE harus 'minmax', 'zscore', atau 'l2'")

# ========= ELBOW METHOD =========
wcss_list = []
diff_list = []

for k in range(K_MIN, K_MAX + 1):
    wcss = kmeans_wcss(Xn, k)
    wcss_list.append(wcss)
    if len(wcss_list) > 1:
        diff_list.append(wcss_list[-2] - wcss_list[-1])
    else:
        diff_list.append(0)  # selisih pertama ga ada

# Buat dataframe hasil
df_wcss = pd.DataFrame({
    "K": list(range(K_MIN, K_MAX + 1)),
    "WCSS": [round(w, 4) for w in wcss_list],
    "Diff": [round(d, 4) for d in diff_list]
})

print("\n=== Hasil WCSS & Selisih ===")
print(df_wcss.to_string(index=False))

# Cari selisih terbesar (abaikan None di index pertama)
max_diff_idx = np.argmax(diff_list[1:]) + 1
K_optimal = df_wcss.iloc[max_diff_idx]["K"]
print(f"\nK optimal (berdasarkan selisih WCSS terbesar) = {int(K_optimal)}")

plt.figure(figsize=(8,5))
plt.plot(range(K_MIN, K_MAX+1), wcss_list, marker='o')
plt.title(f"Elbow Method ({NORMALIZATION_MODE.upper()} Normalization)")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ========= INPUT K OPTIMAL =========
# K_optimal = int(input("Masukkan jumlah cluster optimal (K) dari grafik: "))

# ========= JALANKAN K-MEANS FINAL =========
# labels, centroids, wcss = kmeans_manual_iter(Xn, teams, int(K_optimal), feature_names=df_grouped.columns[1:], verbose=True)

# [NEW] jika INIT_MODE manual, pakai centroid awal dari indeks 1,3,5
if INIT_MODE == "manual":
    init_indices_0based = [i - 1 for i in INIT_INDICES_1BASED]  # konversi 1-based -> 0-based
    K_manual = len(init_indices_0based)
    labels, centroids, wcss = kmeans_manual_iter(
        Xn, teams, K_manual,
        verbose=True,
        init_mode="manual",
        init_indices_0based=init_indices_0based
    )
else:
    labels, centroids, wcss = kmeans_manual_iter(
        Xn, teams, int(K_optimal),
        verbose=True,
        init_mode="kmeans++"
    )

# ========= TAMPILKAN SILHOUETTE SCORE =========
scores = silhouette_scores(Xn, labels)
df_result = pd.DataFrame({
    "Team": teams,
    "Cluster": [lbl+1 for lbl in labels],
    "Silhouette": [round(s, 4) for s in scores]
})
print(df_result.to_string(index=False))

print(f"\nSilhouette Score rata-rata = {sum(scores)/len(scores):.4f}")
