import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =======================================
# 1) KONFIG
# =======================================
FILE_PATH   = "data.xlsx"      # ganti jika perlu
SHEET_NAME  = "FWD"            # atau "complete"
BWS         = np.arange(0.5, 10.5, 0.5)  # 0.5, 1.0, ..., 10.0
# BWS         = [2.5] # 0.5, 1.0, ..., 10.0
# BWS         = np.arange(1, 11, 1)  # 0.5, 1.0, ..., 10.0

# daftar eksplisit yang TIDAK ikut clustering (case-insensitive; akan dicocokkan setelah lower())
EXCLUDE_COLS = [
    'player', 'team', 'nat', 'pos', 'age', 'app',
    'total minutes', '90s', 'total goal', 'assist', 'errors'
    'longball', 'balls recovered/game', 'dribbled past/game', 'acc pass/game'
    'clearances/game', 'errors leading to shot/90s', 'fouls/game'
]
# keyword bantu untuk partial match (mis. "app" nangkep "apps"/"appearances", "minute" nangkep "minutes played")
EXCLUDE_KEYWORDS = [
    
]

# =======================================
# 2) BACA DATA & SIAPKAN FITUR
# =======================================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=0)
orig_cols = df.columns.tolist()
df.columns = [c.strip().lower() for c in df.columns]  # normalisasi

if df.shape[1] < 2:
    raise ValueError("Minimal butuh 2 kolom: [nama pemain] + >=1 kolom fitur.")

# ambil kolom identitas untuk ditampilkan
name_col = next((c for c in df.columns if re.search(r"\b(player|name)\b", c)), df.columns[0])
team_col = next((c for c in df.columns if re.search(r"\b(team|club|tim)\b", c)), None)
pos_col  = next((c for c in df.columns if re.search(r"\bpos(ition)?\b", c)), None)

player_names = df[name_col].astype(str)
teams = df[team_col].astype(str) if team_col else pd.Series(["UNK"] * len(df), index=df.index)
positions = df[pos_col].astype(str) if pos_col else pd.Series(["UNK"] * len(df), index=df.index)

# bangun daftar kolom yang harus di-exclude (exact + partial)
exclude_exact = {c.strip().lower() for c in EXCLUDE_COLS}
def should_exclude(col: str) -> bool:
    if col in exclude_exact:
        return True
    return any(kw in col for kw in EXCLUDE_KEYWORDS)

excluded_cols_found = [c for c in df.columns if should_exclude(c)]
included_candidates = [c for c in df.columns if c not in excluded_cols_found]

# hanya ambil kandidat yang bernilai numerik (atau bisa dikonversi numerik)
X_raw = df[included_candidates].copy()
for c in X_raw.columns:
    if not np.issubdtype(X_raw[c].dtype, np.number):
        s = X_raw[c].astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
        vals = pd.to_numeric(s, errors="coerce")
        # terima kolom kalau memang sebagian ada angka
        if vals.notna().any():
            X_raw[c] = vals
        else:
            # buang kalau sama sekali tidak numerik
            X_raw.drop(columns=[c], inplace=True)

# bersihkan NaN/inf dan kolom kosong
X = X_raw.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    raise ValueError("Tidak ada fitur numerik setelah pembersihan & exclude.")

X = X.fillna(X.median(numeric_only=True))

# ======== BUKTI: cetak kolom yang dipakai & yang dibuang ========
print("\n=== KOLOM DIBUANG (ditemukan di file) ===")
for c in orig_cols:
    cl = c.strip().lower()
    if cl in excluded_cols_found:
        print("-", c)

print("\n=== KOLOM DIPAKAI untuk clustering (setelah cleaning) ===")
for c in X.columns:
    # tampilkan nama asli sesuai di file (bukan lower) jika mau:
    orig = next((oc for oc in orig_cols if oc.strip().lower() == c), c)
    print("-", orig)

n_samples = len(X)
if n_samples < 3:
    raise ValueError("Butuh ≥3 baris untuk evaluasi clustering.")

# =======================================
# 3) STANDARISASI (Z-SCORE)
# =======================================
scaler = RobustScaler()
X_std = scaler.fit_transform(X)

# =======================================
# 4) LOOP BANDWIDTH & EVALUASI
# =======================================
def valid_for_silhouette(k, n):
    return (k >= 2) and (k <= n - 1)


def gaussian_mean_shift_predict(X, bandwidth, max_iter=100, tol=1e-3, merge_thresh=None):
    """
    Mean Shift dengan kernel Gaussian (custom).
    - X: array (n_samples, n_features), sebaiknya sudah distandardisasi (Z-score).
    - bandwidth (h): radius/scale kernel.
    - merge_thresh: ambang penggabungan mode; default 0.3*h.
    Return:
      labels: np.array shape (n_samples,), label 0..k-1
      centers: np.array shape (k, n_features), pusat (mode) tiap cluster
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if merge_thresh is None:
        merge_thresh = 0.3 * float(bandwidth)

    # inisialisasi posisi "bergerak" = titik awal
    Y = X.copy()

    # iterasi geser ke mean tertimbang Gaussian
    for _ in range(max_iter):
        Y_old = Y
        # hitung mean tertimbang untuk setiap titik (O(n^2))
        # dist2[i, j] = ||Y[i] - X[j]||^2
        dist2 = np.sum((Y[:, None, :] - X[None, :, :])**2, axis=2)  # (n, n)
        W = np.exp(-dist2 / (2.0 * (bandwidth**2)))                  # bobot Gaussian
        # normalisasi bobot per baris
        W_sum = W.sum(axis=1, keepdims=True) + 1e-12
        Y = (W @ X) / W_sum

        # cek konvergensi (rata2 pergeseran)
        shift = np.linalg.norm(Y - Y_old) / n
        if shift < tol:
            break

    # gabungkan mode yang berdekatan (clustering mode)
    centers = []
    labels = -np.ones(n, dtype=int)
    for i in range(n):
        yi = Y[i]
        assigned = False
        for c_idx, c in enumerate(centers):
            if np.linalg.norm(yi - c) <= merge_thresh:
                labels[i] = c_idx
                # update center (opsional: incremental mean)
                centers[c_idx] = (centers[c_idx] + yi) / 2.0
                assigned = True
                break
        if not assigned:
            centers.append(yi)
            labels[i] = len(centers) - 1

    centers = np.vstack(centers)

    # rapikan: reindex label jadi 0..k-1 berurutan
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    labels = np.array([remap[l] for l in labels], dtype=int)
    return labels, centers


results = []
invalids = []
best = None
fallback = None

print("\n=== Sweep bandwidth 0.5 .. 10.0 ===")
for bw in BWS:
    ms = MeanShift(bandwidth=float(bw), bin_seeding=True, cluster_all=True)
    # ms = KMeans(init='k-means++', n_clusters= bw)
    labels_try = ms.fit_predict(X_std)
    # labels_try, _ = gaussian_mean_shift_predict(X_std, bandwidth=float(bw), max_iter=100, tol=1e-3, merge_thresh=None)
    k = len(np.unique(labels_try))

    # fallback: pilih K yang tidak ekstrem
    if (fallback is None) or (
        (1 < k < n_samples and (fallback["k"] in (1, n_samples) or k > fallback["k"])) or
        (fallback["k"] == 1 and k > 1) or
        (fallback["k"] == n_samples and k < n_samples)
    ):
        fallback = {"bw": bw, "labels": labels_try, "k": k, "sil": None, "dbi": None}

    if valid_for_silhouette(k, n_samples):
        sil = silhouette_score(X_std, labels_try)
        dbi = davies_bouldin_score(X_std, labels_try)
        results.append((bw, k, sil, dbi))
        cand = {"bw": bw, "labels": labels_try, "k": k, "sil": sil, "dbi": dbi}
        if (best is None) or (cand["sil"] > best["sil"] or (np.isclose(cand["sil"], best["sil"]) and cand["dbi"] < best["dbi"])):  
            best = cand
        print(f"bw={bw:4.1f} -> k={k:2d} | Sil={sil:.3f} | DBI={dbi:.3f}")
    else:
        why = "single cluster" if k == 1 else ("all-singletons" if k == n_samples else "invalid K")
        invalids.append((bw, k, why))
        print(f"bw={bw:4.1f} -> k={k:2d} | ({why})")

# pilih terbaik
if best is None:
    chosen = fallback
    if chosen is None:
        raise RuntimeError("Tidak ada konfigurasi bandwidth yang bisa dipakai. Cek data atau rentang bandwidth.")
    labels = chosen["labels"]; k = chosen["k"]; sil = dbi = None
    print("\nTidak ada bandwidth valid untuk Silhouette/DBI. Menggunakan fallback.")
else:
    chosen = best
    labels = chosen["labels"]; k = chosen["k"]; sil = chosen["sil"]; dbi = chosen["dbi"]

# =======================================
# 5) RINGKASAN HASIL
# =======================================
if results:
    df_res = pd.DataFrame(results, columns=["bandwidth","n_clusters","silhouette","dbi"])\
             .sort_values(by=["silhouette","dbi"], ascending=[False, True])
    print("\n=== Peringkat bandwidth (valid) ===")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
else:
    print("\nTidak ada hasil valid (Sil/DBI). Daftar invalid:")
    for bw, k, why in invalids:
        print(f"- bw={bw:.1f} -> k={k} | {why}")

print("\n=== Konfigurasi Terpilih ===")
print(f"Bandwidth   : {chosen['bw']:.1f}")
print(f"n_clusters  : {k}")
if sil is not None:
    print(f"Silhouette  : {sil:.4f} (semakin tinggi semakin baik)")
    print(f"Davies-Bouldin: {dbi:.4f} (semakin rendah semakin baik)")
else:
    print("Silhouette/DBI: N/A (k=1 atau k=n). Pertimbangkan sesuaikan rentang bandwidth.")

# tampilkan cluster hasil akhir
# print("\n=== Hasil Cluster ===")
# for name, team, pos, lab in zip(player_names, teams, positions, labels):
#     print(f"{name:25s} | {team:15s} | {pos:8s} | Cluster {lab}")

# =======================================
# 6) VISUALISASI PCA 2D
# =======================================
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

plt.figure(figsize=(10,8))
scatter = plt.scatter(Z[:,0], Z[:,1], c=labels, s=70, cmap="viridis", edgecolors="k", linewidths=0.5)

for i, name in enumerate(player_names):
    plt.text(Z[i,0], Z[i,1], name, fontsize=8, alpha=0.75)

title_meta = f"bw={chosen['bw']:.1f}, k={k}"
metrics = f" | Sil={sil:.3f}, DBI={dbi:.3f}" if sil is not None else " | metrics=N/A"
plt.title(f"Mean Shift (PCA 2D) — Z-Score\n{title_meta}{metrics}")
plt.xlabel("PC1"); plt.ylabel("PC2")
handles, _ = scatter.legend_elements()
plt.legend(handles, [f"Cluster {c}" for c in sorted(np.unique(labels))], title="Cluster", loc="best")
plt.tight_layout()
plt.show()
