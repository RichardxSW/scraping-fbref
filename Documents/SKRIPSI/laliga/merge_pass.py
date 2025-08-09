from pathlib import Path
import pandas as pd

folder = Path("laliga")
# === 1. Baca file ===
laliga_df = pd.read_excel("laliga/pass/laliga_updated_pass_laspalmas.xlsx")
barca_for_df = pd.read_csv("laliga/pass/barcelona_pass.csv")
barca_against_df = pd.read_csv("laliga/pass/barcelona_pass_against.csv")

# Pastikan kolom date jadi datetime
laliga_df["date"] = pd.to_datetime(laliga_df["date"], errors="coerce")
barca_for_df["date"] = pd.to_datetime(barca_for_df["date"], errors="coerce")
barca_against_df["date"] = pd.to_datetime(barca_against_df["date"], errors="coerce")

# === 2. Loop semua Valladolid possession (FOR) ===
for _, row in barca_for_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Valladolid sebagai HOME
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Valladolid")
    laliga_df.loc[mask_home, "HTP"] = passes
    laliga_df.loc[mask_home, "HSP"] = success_passes

    # Real Valladolid sebagai AWAY
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Valladolid")
    laliga_df.loc[mask_away, "ATP"] = passes
    laliga_df.loc[mask_away, "ASP"] = success_passes

# === 3. Loop semua Valladolid possession (AGAINST) ===
for _, row in barca_against_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Lawan Valladolid HOME → berarti possession against masuk ke HPoss
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Valladolid")
    laliga_df.loc[mask_home, "HTP"] = passes
    laliga_df.loc[mask_home, "HSP"] = success_passes

    # Lawan Real Valladolid AWAY → possession against masuk ke APoss
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Valladolid")
    laliga_df.loc[mask_away, "ATP"] = passes
    laliga_df.loc[mask_away, "ASP"] = success_passes
 
# === 4. Simpan hasil ===
laliga_df.to_excel("laliga/pass/laliga_updated_pass_valladolid.xlsx", index=False)
print("Data pass Valladolid berhasil dimasukkan.")
