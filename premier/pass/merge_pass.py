from pathlib import Path
import pandas as pd

folder = Path("premier")
# === 1. Baca file ===
premier_df = pd.read_excel("premier/pass/premier_updated_pass_Leicester_City.xlsx")
barca_for_df = pd.read_csv("premier/pass/liverpool_pass.csv")
barca_against_df = pd.read_csv("premier/pass/liverpool_pass_against.csv")

# Pastikan kolom date jadi datetime
premier_df["date"] = pd.to_datetime(premier_df["date"], errors="coerce")
barca_for_df["date"] = pd.to_datetime(barca_for_df["date"], errors="coerce")
barca_against_df["date"] = pd.to_datetime(barca_against_df["date"], errors="coerce")

# === 2. Loop semua Ipswich Town possession (FOR) ===
for _, row in barca_for_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Ipswich Town sebagai HOME
    mask_home = (premier_df["date"] == match_date) & (premier_df["HomeTeam"] == "Ipswich Town")
    premier_df.loc[mask_home, "HTP"] = passes
    premier_df.loc[mask_home, "HSP"] = success_passes

    # Real Ipswich Town sebagai AWAY
    mask_away = (premier_df["date"] == match_date) & (premier_df["AwayTeam"] == "Ipswich Town")
    premier_df.loc[mask_away, "ATP"] = passes
    premier_df.loc[mask_away, "ASP"] = success_passes

# === 3. Loop semua Ipswich Town possession (AGAINST) ===
for _, row in barca_against_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Lawan Ipswich Town HOME → berarti possession against masuk ke HPoss
    mask_home = (premier_df["date"] == match_date) & (premier_df["AwayTeam"] == "Ipswich Town")
    premier_df.loc[mask_home, "HTP"] = passes
    premier_df.loc[mask_home, "HSP"] = success_passes

    # Lawan Real Ipswich Town AWAY → possession against masuk ke APoss
    mask_away = (premier_df["date"] == match_date) & (premier_df["HomeTeam"] == "Ipswich Town")
    premier_df.loc[mask_away, "ATP"] = passes
    premier_df.loc[mask_away, "ASP"] = success_passes
 
# === 4. Simpan hasil ===
premier_df.to_excel("premier/pass/premier_updated_pass_Ipswich_Town.xlsx", index=False)
print("Data pass Ipswich Town berhasil dimasukkan.")
