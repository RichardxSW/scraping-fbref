import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

# =======================================
# 1) KONFIG
# =======================================
FILE_PATH  = "data_pemain.xlsx"
SHEET_NAME = "data"
BANDWIDTH_LIST = list(range(1, 15))  # 1..10

# =======================================
# 2) BACA DATA
# =======================================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=0)

# kolom pertama = nama
player_names = df.iloc[:, 0].astype(str)

# coba temukan kolom posisi
pos_candidates = [c for c in df.columns if c.lower() in
                  {"pos","position","positions","role","roles","jabatan","peran"}]
pos_col = pos_candidates[0] if pos_candidates else None
if pos_col is None:
    print("PERINGATAN: Kolom posisi tidak ditemukan. Lanjut tanpa stratifikasi posisi.")
    pos_series = pd.Series(["UNK"] * len(df))
else:
    pos_series = df[pos_col].astype(str)

# map teks posisi ke grup makro: GK / DEF / MID / FWD
def map_pos(x: str) -> str:
    s = x.strip().upper()
    # Inggris / kode
    if re.search(r"\bGK|GOAL|KEEP", s): return "GK"
    if re.search(r"\bCB|LB|RB|CB|DEF|BACK|WINGBACK|WB", s): return "DEF"
    if re.search(r"\bDM|CM|AM|DM|RM|LM|MID|MIDFIEL", s): return "MID"
    if re.search(r"\bFW|ST|CF|LW|RW|WINGER|STRIK", s): return "FWD"
    # Indonesia
    if "KIPER" in s: return "GK"
    if any(w in s for w in ["BEK","STOPPER","FULLBACK","SAYAP BERTAHAN"]): return "DEF"
    if "GELANDANG" in s: return "MID"
    if any(w in s for w in ["PENYERANG","STRIKER","SAYAP"]): return "FWD"
    return "UNK"

pos_group = pos_series.map(map_pos)

# ambil fitur numerik (kecuali kolom pertama)
X_raw = df.iloc[:, 1:].copy()
for c in X_raw.columns:
    if not np.issubdtype(X_raw[c].dtype, np.number):
        s = X_raw[c].astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
        X_raw[c] = pd.to_numeric(s, errors="coerce")

X = X_raw.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    raise ValueError("Tidak ada fitur numerik setelah pembersihan.")
X = X.fillna(X.median(numeric_only=True))  # isi NaN sisanya

# =======================================
# 3) Z-SCORE **PER POSISI**
# =======================================
# skala setiap fitur DI DALAM masing-masing grup posisi → fokus ke pola/gaya
X_std = np.zeros_like(X.values, dtype=float)
X_std = pd.DataFrame(X_std, columns=X.columns, index=X.index)

for grp, idx in pos_group.groupby(pos_group).groups.items():
    scaler = StandardScaler()
    X_std.loc[idx, :] = scaler.fit_transform(X.loc[idx, :])

X_std = X_std.values
n_samples = X_std.shape[0]
if n_samples < 3:
    raise ValueError("Butuh ≥3 baris untuk evaluasi clustering.")

# =======================================
# 4) SWEEP BANDWIDTH 1..10
# =======================================
def valid_for_silhouette(k, n):
    return (k >= 2) and (k <= n - 1)

best = None
fallback = None
records = []

print("=== Sweep bandwidth 1..10 (z-score per posisi) ===")
for bw in BANDWIDTH_LIST:
    ms = MeanShift(bandwidth=float(bw), bin_seeding=True, cluster_all=True)
    labels_try = ms.fit_predict(X_std)
    k = len(np.unique(labels_try))

    # fallback: prefer K di tengah (bukan 1, bukan n); kalau tidak ada, ambil K terbesar < n
    if (fallback is None) or (
        (1 < k < n_samples and (fallback["k"] in (1, n_samples) or k > fallback["k"])) or
        (fallback["k"] == 1 and k > 1) or
        (fallback["k"] == n_samples and k < n_samples)
    ):
        fallback = {"bw": bw, "labels": labels_try, "k": k, "sil": None, "dbi": None}

    if valid_for_silhouette(k, n_samples):
        sil = silhouette_score(X_std, labels_try)
        dbi = davies_bouldin_score(X_std, labels_try)
        records.append((bw, k, sil, dbi))
        cand = {"bw": bw, "labels": labels_try, "k": k, "sil": sil, "dbi": dbi}
        if (best is None) or (cand["sil"] > best["sil"] or (np.isclose(cand["sil"], best["sil"]) and cand["dbi"] < best["dbi"])):
            best = cand
        print(f"bw={bw:2d} -> k={k:2d} | Sil={sil:.3f} | DBI={dbi:.3f}")
    else:
        why = "single cluster" if k == 1 else ("all-singletons" if k == n_samples else "invalid K")
        print(f"bw={bw:2d} -> k={k:2d} | ({why})")

# pilih terbaik
if best is None:
    chosen = fallback
    labels = chosen["labels"]; k = chosen["k"]; sil = dbi = None
    print("\nTidak ada bandwidth valid untuk Sil/DBI. Pakai fallback.")
else:
    chosen = best
    labels = chosen["labels"]; k = chosen["k"]; sil = chosen["sil"]; dbi = chosen["dbi"]

# rangkum tabel hasil (kalau ada)
if records:
    df_res = pd.DataFrame(records, columns=["bandwidth","n_clusters","silhouette","dbi"])\
             .sort_values(by=["silhouette","dbi"], ascending=[False, True])
    print("\n=== Peringkat bandwidth (valid) ===")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== Konfigurasi Terpilih ===")
print(f"Bandwidth : {chosen['bw']}")
print(f"k (clusters): {k}")
if sil is not None:
    print(f"Silhouette: {sil:.4f}  | DBI: {dbi:.4f}")

# Crosstab Cluster × Posisi
ct = pd.crosstab(pd.Series(labels, name="Cluster"), pd.Series(pos_group, name="Pos"))
print("\n=== Cluster × Posisi ===")
print(ct.to_string())

# =======================================
# 5) VISUALISASI: warna=cluster, marker=posisi
# =======================================
pca = PCA(n_components=2, svd_solver='full')
Z = pca.fit_transform(X_std)

markers = {"GK":"s", "DEF":"^", "MID":"o", "FWD":"D", "UNK":"x"}
plt.figure(figsize=(10,8))

for grp in sorted(pos_group.unique()):
    m = markers.get(grp, "x")
    idx = (pos_group == grp)
    plt.scatter(Z[idx,0], Z[idx,1], c=np.array(labels)[idx], s=70, cmap="viridis", marker=m, label=f"{grp}")

for i, name in enumerate(player_names):
    plt.text(Z[i,0], Z[i,1], name, fontsize=8, alpha=0.7)

plt.title(f"Mean Shift (PCA 2D)\nZ-Score per Posisi | bw={chosen['bw']}, k={k}"
          + (f" | Sil={sil:.3f}, DBI={dbi:.3f}" if sil is not None else ""))
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(title="Posisi (marker)", loc="best")
plt.tight_layout()
plt.show()
