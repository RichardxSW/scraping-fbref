import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import numpy as np

# ========= PARAMETER =========
FILE_PATH = "data_laliga.xlsx"   # ganti sesuai file Excel
SKIP_TOP_ROWS = 0
K_MIN, K_MAX = 2, 10           # range K untuk elbow method
MAX_ITER = 200
TOL = 1e-6
NORMALIZATION_MODE = "minmax"     # "minmax", "zscore", atau "l2"

# ========= FUNGSI UTIL =========
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def compute_wcss(X, labels, centroids):
    return sum(euclidean(X[i], centroids[labels[i]])**2 for i in range(len(X)))

# ========= KMEANS UTAMA =========
def kmeans_manual_iter(X, teams, k, max_iter=MAX_ITER, tol=TOL, verbose=True):
    n = len(X)
    # init_indices = list(range(k))
    init_indices = random.sample(range(n), k)
    centroids = [X[i] for i in init_indices]
    labels_old = [-1] * n

    for it in range(1, max_iter + 1):
        labels = []
        clusters = [[] for _ in range(k)]
        for x in X:
            dists = [euclidean(x, c) for c in centroids]
            cid = dists.index(min(dists))
            labels.append(cid)
            clusters[cid].append(x)

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
            df_iter = pd.DataFrame(X, columns=[f"Feature{i+1}" for i in range(X.shape[1])])
            if teams is not None:
                df_iter["Team"] = teams
            df_iter["Cluster"] = [lbl+1 for lbl in labels]
            print(df_iter.to_string(index=False))

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

# ========= BACA & NORMALISASI DATA =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS)
teams = df.iloc[:, 0].astype(str)
num = df.iloc[:, 1:11].apply(pd.to_numeric, errors='coerce')
mask = num.notnull().all(axis=1)
teams = teams[mask].reset_index(drop=True)
X = num[mask].to_numpy()

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

else:
    raise ValueError("NORMALIZATION_MODE harus 'minmax', 'zscore', atau 'l2'")

# ========= ELBOW METHOD =========
wcss_list = []
for k in range(K_MIN, K_MAX + 1):
    wcss = kmeans_wcss(Xn, k)
    wcss_list.append(wcss)

plt.figure(figsize=(8,5))
plt.plot(range(K_MIN, K_MAX+1), wcss_list, marker='o')
plt.title(f"Elbow Method ({NORMALIZATION_MODE.upper()} Normalization)")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ========= INPUT K OPTIMAL =========
K_optimal = int(input("Masukkan jumlah cluster optimal (K) dari grafik: "))

# ========= JALANKAN K-MEANS FINAL =========
labels, centroids, wcss = kmeans_manual_iter(Xn, teams, K_optimal, verbose=True)

# ========= TAMPILKAN SILHOUETTE SCORE =========
scores = silhouette_scores(Xn, labels)
df_result = pd.DataFrame({
    "Team": teams,
    "Cluster": [lbl+1 for lbl in labels],
    "Silhouette": [round(s, 4) for s in scores]
})
print(df_result.to_string(index=False))

print(f"\nSilhouette Score rata-rata = {sum(scores)/len(scores):.4f}")
