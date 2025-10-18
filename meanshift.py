import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==============================
# KONFIG
# ==============================
FILE_PATH    = "data.xlsx"      # ganti jika perlu
SHEET_NAME   = "ALL"            # atau "complete"
BWS          = np.arange(0.5, 10.5, 0.5)  # 0.5, 1.0, ..., 10.0
# BWS          = np.arange(1, 11, 1)  # 0.5, 1.0, ..., 10.0
# BWS        = [1.5]
MIN_MINUTES  = 0

# ==============================
# 1) BACA DATA
# ==============================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=0)
orig_cols = df.columns.tolist()
df.columns = [c.strip().lower() for c in df.columns]

if df.shape[1] < 2:
    raise ValueError("Minimal butuh 2 kolom: [nama pemain] + >=1 kolom fitur.")

# ==============================
# 1a) FILTER: menit > MIN_MINUTES
# deteksi kolom menit otomatis; fallback pakai '90s' * 90
# ==============================
MINUTE_CANDIDATES = [
    "total minute"
]
minute_col = None
for c in df.columns:
    if any(k in c for k in MINUTE_CANDIDATES):
        minute_col = c
        break

nineties_col = next((c for c in df.columns if re.fullmatch(r"\s*90s\s*", c)), None)

if minute_col is not None:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
elif nineties_col is not None:
    minute_col = "__minutes_from_90s__"
    df[minute_col] = pd.to_numeric(df[nineties_col], errors="coerce") * 90
else:
    raise ValueError("Tidak ada kolom menit maupun '90s' untuk memfilter pemain.")

before_n = len(df)
df = df[df[minute_col] > MIN_MINUTES].copy()
after_n = len(df)
print(f"\nFilter menit: > {MIN_MINUTES}. Tersisa {after_n}/{before_n} pemain.")
if after_n < 3:
    raise ValueError("Setelah filter menit, baris < 3. Longgarkan MIN_MINUTES atau cek data.")

# ==============================
# 1b) KOLOM IDENTITAS
# ==============================
name_col = next((c for c in df.columns if re.search(r"\b(player|name)\b", c)), df.columns[0])
team_col = next((c for c in df.columns if re.search(r"\b(team|club|tim)\b", c)), None)
pos_col  = next((c for c in df.columns if re.search(r"\bpos(ition)?\b", c)), None)

player_names_full = df[name_col].astype(str)
teams_full = df[team_col].astype(str) if team_col else pd.Series(["UNK"] * len(df), index=df.index)
positions_full = df[pos_col].astype(str) if pos_col else pd.Series(["UNK"] * len(df), index=df.index)

# ==============================
# 2) POSISI: SPLIT SEMUA TOKEN
# (_SPLIT_PAT + alias split_plat)
# ==============================
_SPLIT_PAT = re.compile(r"[,\s/\|\-]+")   # koma, spasi, '/', '|', '-'
split_plat = _SPLIT_PAT                   # alias kalau kamu refer dgn nama ini

def split_pos_tokens(p: str) -> list[str]:
    p = (p or "").strip().lower()
    if not p:
        return []
    toks = [t for t in _SPLIT_PAT.split(p) if t]
    return toks

# map index -> set token posisi
pos_tokens_map = {idx: set(split_pos_tokens(val)) for idx, val in positions_full.items()}
uniq_pos_all = sorted(set(t for toks in pos_tokens_map.values() for t in toks)) or ["unk"]
print(f"\n=== Posisi unik terdeteksi: {uniq_pos_all}")

# ==============================
# 3) GROUPING POSISI
# ==============================
STRIKER_POS = {"st"}      
WING_POS = {"lw", "rw", "lm", "rm"}
MID_POS = {"cm", "dm"}
DEF_POS = {"cb"}
WINGDEF_POS = {"lb", "rb"}

GROUPS = [
    ("ST", STRIKER_POS),
    ("WING", WING_POS),
    ("MID", MID_POS),
    ("DEF", DEF_POS),
    ("WINGDEF", WINGDEF_POS),
]

# ==============================
# 4) FEATURES (HARD-CODE PER GRUP)
# isi dgn header persis dari Excel (case & spasi fleksibel krn dinormalisasi)
# ==============================
FEATURES = {
    "ST": [
        # --- CONTOH: ganti sesuai sheet kamu ---
        "goal/game",
        "assist/game",
        "sot/game",
        # "shot/game",
        "aerial won/game",
        "total duel won/game",
        "s. dribble/game "
        # "key pass/game",
        # "acc pass/game",
    ],
    "WING": [
        "s. dribble/game",
        "acc cross/game",
        "total duel won/game",
        # "aerial won/game",
        "key pass/game",
        "sot/game",
        "shot/game"
    ],
    "MID": [
        "key pass/game",
        "s. dribble/game",
        "assist/game",        
        "acc pass/game",
        "long ball/game",
        "ball recovered/game",
        "total duel won/game"
    ],
    "DEF": [
        "long ball/game",
        "ball recovered/game",
        "clearance/game",
        "error leading to shot/game",
        "aerial won/game",
        # "total duel won/game",
    ],
    "WINGDEF": [
        "acc cross/game",
        "long ball/game",
        "ball recovered/game",
        "clearance/game",
        "error leading to shot/game",
        "total duel won/game"
    ]
}

def _normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s*/\s*", "/", s)   # 'pass / game' -> 'pass/game'
    s = re.sub(r"\s+", " ", s)
    return s

def resolve_columns(hard_list: list[str], df_cols: list[str]) -> tuple[list[str], list[str]]:
    norm_map = {_normalize_name(c): c for c in df_cols}  # normal->original
    picked, missing = [], []
    for want in hard_list:
        key = _normalize_name(want)
        if key in norm_map:
            picked.append(norm_map[key])
        else:
            missing.append(want)
    return picked, missing

# ==============================
# 5) UTIL
# ==============================
def to_numeric_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    df_num = df_in.copy()
    for c in df_num.columns:
        if c in {name_col, team_col, pos_col}:
            continue
        if not np.issubdtype(df_num[c].dtype, np.number):
            s = df_num[c].astype(str).str.replace(",", ".", regex=False)
            s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
            vals = pd.to_numeric(s, errors="coerce")
            df_num[c] = vals
    return df_num

def choose_feature_columns(df_num: pd.DataFrame, group_name: str) -> list[str]:
    hard = FEATURES.get(group_name, [])
    if not hard:
        print(f"[{group_name}] WARNING: FEATURES kosong. Grup di-skip.")
        return []
    numeric_cols = [c for c in df_num.columns if np.issubdtype(df_num[c].dtype, np.number)]
    # buang identitas jika numeric
    for idc in [name_col, team_col, pos_col]:
        if idc in numeric_cols:
            numeric_cols.remove(idc)
    feat_cols, missing = resolve_columns(hard, numeric_cols)
    if missing:
        print(f"[{group_name}] Kolom TIDAK ditemukan (diabaikan): {missing}")
    if not feat_cols:
        print(f"[{group_name}] Skip: semua kolom hardcode tidak ada di sheet.")
        return []
    return feat_cols

def valid_for_silhouette(k, n):
    return (k >= 2) and (k <= n - 1)

def run_meanshift_sweep(X_std: np.ndarray, BWS: np.ndarray):
    n_samples = len(X_std)
    best = None
    fallback = None
    results = []
    invalids = []

    print("\n=== Sweep bandwidth ===")
    for bw in BWS:
        ms = MeanShift(bandwidth=float(bw), bin_seeding=True, cluster_all=True)
        # ms = KMeans(init="k-means++", n_clusters=bw)
        labels_try = ms.fit_predict(X_std)
        k = len(np.unique(labels_try))

        # fallback: pilih K tidak ekstrem
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

    if best is None:
        chosen = fallback
        if chosen is None:
            raise RuntimeError("Tidak ada konfigurasi bandwidth yang bisa dipakai.")
        labels = chosen["labels"]; k = chosen["k"]; sil = dbi = None
    else:
        chosen = best
        labels = chosen["labels"]; k = chosen["k"]; sil = chosen["sil"]; dbi = chosen["dbi"]

    return chosen, labels, k, sil, dbi, results, invalids

def plot_pca_scatter(X_std: np.ndarray, labels: np.ndarray, names: pd.Series, title: str, metrics_str: str):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_std)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=labels, s=70, cmap="viridis", edgecolors="k", linewidths=0.5)
    for i, name in enumerate(names):
        plt.text(Z[i,0], Z[i,1], str(name), fontsize=8, alpha=0.75)
    plt.title(f"{title}\n{metrics_str}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {c}" for c in sorted(np.unique(labels))], title="Cluster", loc="best")
    plt.tight_layout()
    plt.show()

# ==============================
# 6) LOOP PER GRUP (multi-posisi OK)
# ==============================
for group_name, pos_set in GROUPS:
    # mask: baris yang punya irisan token posisi vs set grup
    mask = pd.Series(df.index).apply(
        lambda idx: len(pos_tokens_map.get(idx, set()) & pos_set) > 0
    ).values

    print(f"\n[{group_name}] tokens target: {sorted(pos_set)}")
    print(f"[{group_name}] matched rows: {int(mask.sum())}")

    df_g = df.loc[mask].copy()
    if df_g.empty or df_g.shape[0] < 3:
        print(f"[{group_name}] Skip: baris < 3 atau tidak ada pemain di grup ini.")
        continue

    names_g = df_g[name_col].astype(str)

    # numerikkan & pilih fitur
    df_num = to_numeric_frame(df_g)
    feature_cols = choose_feature_columns(df_num, group_name)
    if not feature_cols:
        print(f"[{group_name}] Skip: tidak ada fitur hardcode yang valid.")
        continue

    X_raw = df_num[feature_cols].copy()

    # cleaning + imputasi
    X = X_raw.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if X.shape[1] == 0:
        print(f"[{group_name}] Skip: tidak ada fitur numerik setelah cleaning.")
        continue
    X = X.fillna(X.median(numeric_only=True))

    print(f"[{group_name}] n_pemain={len(X)} | n_fitur={X.shape[1]}")
    print(f"[{group_name}] Kolom fitur terpakai (HARD-CODE):")
    for c in X.columns:
        orig = next((oc for oc in orig_cols if oc.strip().lower() == c), c)
        print("-", orig)

    # scaling
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # sweep bandwidth
    chosen, labels, k, sil, dbi, results, invalids = run_meanshift_sweep(X_std, BWS)

    # ranking
    val_rows = [(bw, kk, ss, dd) for (bw, kk, ss, dd) in results]
    if val_rows:
        df_res = pd.DataFrame(val_rows, columns=["bandwidth","n_clusters","silhouette","dbi"])\
                 .sort_values(by=["silhouette","dbi"], ascending=[False, True])
        print(f"\n[{group_name}] === Peringkat bandwidth (valid) ===")
        print(df_res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print(f"\n[{group_name}] Tidak ada hasil valid untuk Silhouette/DBI. Daftar invalid:")
        for bw, kk, why in invalids:
            print(f"- bw={bw:.1f} -> k={kk} | {why}")

    # ringkasan
    print(f"\n[{group_name}] === Konfigurasi Terpilih ===")
    print(f"Bandwidth   : {chosen['bw']:.1f}")
    print(f"n_clusters  : {k}")
    if sil is not None:
        print(f"Silhouette  : {sil:.4f} (lebih tinggi lebih baik)")
        print(f"Davies-Bouldin: {dbi:.4f} (lebih rendah lebih baik)")
        metrics_str = f"{group_name} | bw={chosen['bw']:.1f}, k={k} | Sil={sil:.3f}, DBI={dbi:.3f}"
    else:
        print("Silhouette/DBI: N/A (k=1 atau k=n).")
        metrics_str = f"{group_name} | bw={chosen['bw']:.1f}, k={k} | metrics=N/A"

    # plot
    plot_pca_scatter(X_std, labels, names_g, f"Mean Shift (PCA 2D) — {group_name}", metrics_str)
