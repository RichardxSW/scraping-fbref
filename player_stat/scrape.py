import re
import io
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ====== KONFIGURASI ======
URL = "https://fbref.com/en/squads/822bd0ba/2024-2025/Liverpool-Stats"

# Pilih tabel yang ingin diambil. Bisa pakai keyword id atau judul (case-insensitive).
# Contoh umum: "standard", "shooting", "passing", "passing_types", "gca", "defense", "possession", "misc"
TARGET_TABLE_KEYWORDS = [
    "standard", 
    "keeper",
    "shooting", 
    "passing", 
    "defense", 
    "possession", 
    "misc"
]

# Nama file output Excel
OUTFILE = "premier.xlsx"

# User-Agent yang “ramah”
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# ====== UTIL ======
def flatten_columns(cols: pd.MultiIndex | pd.Index) -> List[str]:
    """Ratakan MultiIndex header menjadi satu baris nama kolom yang bersih."""
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            # buang "Unnamed: ..." dan gabungkan dengan underscore
            parts = [str(x).strip() for x in tup if x and not str(x).startswith("Unnamed")]
            flat.append("_".join(parts) if parts else "")
        return [re.sub(r"\s+", " ", c).strip("_ ").replace("\n", " ") for c in flat]
    return [str(c).strip() for c in cols]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan header ganda dan baris header yang nyelip di tengah (mis. 'Player')."""
    df = df.copy()
    # Ratakan kolom
    df.columns = flatten_columns(df.columns)

    # Hapus baris yang merupakan header ulang (sering kolom 'Player' muncul sebagai nilai)
    # Aturan: jika ada kolom 'Player' dan nilainya persis 'Player', drop baris tsb
    if "Player" in df.columns:
        df = df[df["Player"].astype(str).str.lower() != "player"]

    # Drop kolom kosong total
    df = df.dropna(axis=1, how="all")

    # Reset index
    return df.reset_index(drop=True)

# ====== CORE ======
def fetch_html(url: str) -> str:
    """Ambil HTML dengan retry ringan."""
    for i in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.text:
            return resp.text
        time.sleep(2)
    resp.raise_for_status()

def extract_tables(html: str) -> List[Tuple[str, str, str]]:
    """
    Kembalikan list tuple (table_id, data_title, table_html_string).
    FBref sering menyimpan <table> di dalam HTML comments; kita buka semuanya.
    """
    soup = BeautifulSoup(html, "lxml")

    # Buka semua comments yang memuat <table>
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in c or "<TABLE" in c:
            # parse kembali isi comment dan masukkan ke DOM
            frag = BeautifulSoup(c, "lxml")
            for t in frag.find_all("table"):
                soup.append(t)

    tables = []
    for t in soup.find_all("table"):
        tid = (t.get("id") or "").strip()
        # data-title sering berisi judul manusiawi tabel
        dtitle = (t.get("data-title") or t.get("aria-label") or "").strip()
        # hanya tabel statistik (biasanya prefiks 'stats_'), tapi kita ambil semua untuk jaga-jaga
        if tid:
            tables.append((tid, dtitle, str(t)))
    return tables

def filter_tables(tables: List[Tuple[str, str, str]], keywords: List[str]) -> List[Tuple[str, str, str]]:
    if not keywords:
        return tables
    kws = [k.lower() for k in keywords]
    sel = []
    for tid, dtitle, thtml in tables:
        hay = f"{tid} {dtitle}".lower()
        if any(k in hay for k in kws):
            sel.append((tid, dtitle, thtml))
    return sel

def read_table_to_df(table_html: str) -> pd.DataFrame:
    # Gunakan pandas.read_html pada string table-only
    dfs = pd.read_html(io.StringIO(table_html), flavor="lxml")
    if not dfs:
        return pd.DataFrame()
    df = dfs[0]
    return clean_df(df)

def scrape_fbref_tables(url: str, target_keywords: List[str]) -> Dict[str, Tuple[str, pd.DataFrame]]:
    html = fetch_html(url)
    all_tables = extract_tables(html)

    # Tampilkan daftar semua tabel yang tersedia untuk referensi
    print("=== Tabel tersedia di halaman (id  •  title) ===")
    for tid, title, _ in all_tables:
        if tid.startswith("stats_"):
            print(f"{tid:<28} • {title}")

    # Filter hanya yang diinginkan
    selected = filter_tables(all_tables, target_keywords)
    if not selected:
        print("\nTidak ada tabel yang cocok dengan keywords:", target_keywords)
        return {}

    results: Dict[str, Tuple[str, pd.DataFrame]] = {}
    for tid, title, thtml in selected:
        try:
            df = read_table_to_df(thtml)
            if df.empty:
                print(f"[WARN] {tid} ({title}) kosong/ga terbaca")
                continue
            results[tid] = (title, df)
        except Exception as e:
            print(f"[ERROR] gagal parse {tid} ({title}): {e}")
    return results

def save_to_excel(tables: Dict[str, Tuple[str, pd.DataFrame]], outfile: str):
    if not tables:
        print("Tidak ada tabel untuk disimpan.")
        return
    with pd.ExcelWriter(outfile, engine="xlsxwriter") as writer:
        for tid, (title, df) in tables.items():
            # Nama sheet <= 31 char; fallback ke id jika judul kepanjangan/kosong
            sheet = (title or tid).strip()[:31] if (title or "").strip() else tid[:31]
            # Hindari sheet name kosong/aneh
            if not sheet:
                sheet = tid[:31] or "Sheet"
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Selesai. Tersimpan ke: {outfile}")

# ====== JALANKAN ======
if __name__ == "__main__":
    print(f"Ambil dari: {URL}")
    data = scrape_fbref_tables(URL, TARGET_TABLE_KEYWORDS)
    save_to_excel(data, OUTFILE)
