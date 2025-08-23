import pandas as pd
import random, math

# ========= PARAMETER =========
FILE_PATH = "data_tes.xlsx"   # ganti sesuai nama file
SKIP_TOP_ROWS = 0   # kalau ada SATU baris judul di atas header, ubah ke 1
K_MIN, K_MAX = 2, 6
# random.seed()

# ========= FUNGSI UTIL =========
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def compute_wcss(X, labels, centroids):
    wcss = 0.0
    for i, x in enumerate(X):
        c = centroids[labels[i]]
        # jarak kuadrat
        wcss += euclidean(x, c)**2
    return wcss

def kmeans_manual(X, k, max_iter=200, tol=1e-6):
    n = len(X)
    # inisialisasi centroid dari sampel acak
    centroids = [X[i] for i in random.sample(range(n), k)]

    for _ in range(max_iter):
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
        for c in clusters:
            if c:
                dim = len(c[0])
                new_centroids.append([sum(row[j] for row in c)/len(c) for j in range(dim)])
            else:
                # jika cluster kosong, reinit acak
                new_centroids.append(list(X[random.randrange(n)]))

        # cek konvergensi (perubahan total)
        shift = sum(euclidean(centroids[i], new_centroids[i]) for i in range(k))
        centroids = new_centroids
        if shift < tol:
            break

    wcss = compute_wcss(X, labels, centroids)
    return labels, centroids, wcss

# ========= BACA & SIAPKAN DATA =========
df = pd.read_excel(FILE_PATH, header=0, skiprows=SKIP_TOP_ROWS)

# kolom pertama: nama tim (ID), kolom 2–12: fitur numerik
teams = df.iloc[:, 0].astype(str)
num = df.iloc[:, 1:11].apply(pd.to_numeric, errors='coerce')

# buang baris yang tidak lengkap
mask = num.notnull().all(axis=1)
teams = teams[mask].reset_index(drop=True)
X = num[mask].to_numpy()

# normalisasi min–max
mins = X.min(axis=0)
maxs = X.max(axis=0)
den = (maxs - mins)
den[den == 0] = 1.0  # hindari bagi nol untuk fitur konstan
Xn = (X - mins) / den

# ========= PRINT DATA =========
print("=== Nama Tim ===")
print(teams)
print("\n=== Fitur numerik (sebelum normalisasi) ===")
print(X)
print("\n=== Fitur numerik (setelah normalisasi) ===")
print(Xn)

# ========= JALANKAN K-MEANS & CETAK WCSS =========
for k in range(K_MIN, K_MAX + 1):
    labels, centroids, wcss = kmeans_manual(Xn, k)
    print(f"K = {k}, WCSS = {wcss:.6f}")
