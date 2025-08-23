import pandas as pd
import math

# ========= PARAMETER =========
FILE_PATH = "data_tes.xlsx"   # ganti sesuai nama file
SKIP_TOP_ROWS = 0
K = 4  # jumlah cluster
INITIAL_CENTROIDS = [0, 1, 2, 3]  # indeks tim awal sebagai centroid

# ========= FUNGSI UTIL =========
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def compute_wcss(X, labels, centroids):
    wcss = 0.0
    for i, x in enumerate(X):
        c = centroids[labels[i]]
        wcss += euclidean(x, c)**2
    return wcss

# ========= K-MEANS DENGAN PRINT RAPI =========
def kmeans_manual_table(X, teams, k, init_indices=None, max_iter=200, tol=1e-6):
    n = len(X)
    
    # centroid awal
    if init_indices:
        centroids = [X[i] for i in init_indices]
    else:
        import random
        centroids = [X[i] for i in random.sample(range(n), k)]
    
    labels_old = [-1] * n  # label awal
    for it in range(1, max_iter + 1):
        # assign cluster
        labels = []
        clusters = [[] for _ in range(k)]
        for x in X:
            dists = [euclidean(x, c) for c in centroids]
            cid = dists.index(min(dists))
            labels.append(cid)
            clusters[cid].append(x)
        
        # update centroid
        new_centroids = []
        for idx, c in enumerate(clusters):
            if c:
                dim = len(c[0])
                new_centroids.append([sum(row[j] for row in c)/len(c) for j in range(dim)])
            else:
                import random
                new_centroids.append(list(X[random.randrange(n)]))
        
        wcss = compute_wcss(X, labels, new_centroids)
        
        # ===== PRINT RAPI =====
        print(f"\n=== Iterasi {it} ===")
        df_iter = pd.DataFrame(X, columns=[f"Feature{i+1}" for i in range(X.shape[1])])
        df_iter["Team"] = teams
        df_iter["Cluster"] = [lbl + 1 for lbl in labels]
        print(df_iter.to_string(index=False))
        
        print("\nCentroids:")
        for idx, c in enumerate(new_centroids):
            rounded = [round(float(v), 4) for v in c]  
            print(f"Centroid {idx + 1}: {rounded}")
        print(f"\nWCSS = {wcss:.6f}")
        
        # cek konvergensi
        if labels == labels_old:
            print("\nCluster tidak berubah lagi. Iterasi selesai.")
            break
        labels_old = labels
        centroids = new_centroids
        
    return labels, centroids, wcss

# ========= BACA & SIAPKAN DATA =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS)
teams = df.iloc[:, 0].astype(str)
num = df.iloc[:, 1:11].apply(pd.to_numeric, errors='coerce')

mask = num.notnull().all(axis=1)
teams = teams[mask].reset_index(drop=True)
X = num[mask].to_numpy()

# normalisasi min–max
mins = X.min(axis=0)
maxs = X.max(axis=0)
den = (maxs - mins)
den[den == 0] = 1.0
Xn = (X - mins) / den

# ========= JALANKAN K-MEANS =========
labels, centroids, wcss = kmeans_manual_table(Xn, teams, K, init_indices=INITIAL_CENTROIDS)
