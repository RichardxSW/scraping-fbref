import pandas as pd

# 1. Load data Excel kamu
df_excel = pd.read_excel("laliga2425.xlsx")  # ganti nama file sesuai punya kamu

# 2. Load data hasil scraping FBref
df_fbref = pd.read_csv("laliga_xg.csv")


# 3. Normalisasi nama tim supaya matching (hapus spasi ekstra, huruf kecil semua)
def normalize_team(name):
    if pd.isna(name):
        return None
    return str(name).strip().lower()

df_excel["home_norm"] = df_excel["HomeTeam"].apply(normalize_team)
df_excel["away_norm"] = df_excel["AwayTeam"].apply(normalize_team)

df_fbref["home_norm"] = df_fbref["home_team"].apply(normalize_team)
df_fbref["away_norm"] = df_fbref["away_team"].apply(normalize_team)

# 4. Merge berdasarkan home & away team
df_merged = df_excel.merge(
    df_fbref[["home_norm", "away_norm", "hxg", "axg"]],
    on=["home_norm", "away_norm"],
    how="left"
)

# 5. Isi kolom HxG & AxG di Excel kamu
df_merged["HxG"] = df_merged["hxg"]
df_merged["AxG"] = df_merged["axg"]

# 6. Hapus kolom bantu
df_merged = df_merged.drop(columns=["home_norm", "away_norm", "hxg", "axg"])

# 7. Simpan kembali ke Excel
df_merged.to_excel("laliga_updated.xlsx", index=False)

print("Selesai! Data sudah disimpan ke data_football_uk_with_xg.xlsx")
