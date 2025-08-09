from pathlib import Path
import pandas as pd

folder = Path("laliga")
# === 1. Baca file ===
laliga_df = pd.read_excel("laliga_updated_poss_las_palmas.xlsx")
barca_for_df = pd.read_csv("barcelona_possession_tes.csv")
barca_against_df = pd.read_csv("barcelona_possession_against.csv")

# Pastikan kolom date jadi datetime
laliga_df["date"] = pd.to_datetime(laliga_df["date"], errors="coerce")
barca_for_df["date"] = pd.to_datetime(barca_for_df["date"], errors="coerce")
barca_against_df["date"] = pd.to_datetime(barca_against_df["date"], errors="coerce")

# === 2. Loop semua barcelona possession (FOR) ===
for _, row in barca_for_df.iterrows():
    match_date = row["date"]
    poss = row["possession"]

    # Barcelona sebagai HOME
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Valladolid")
    laliga_df.loc[mask_home, "HPoss"] = poss

    # Real Valladolid sebagai AWAY
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Valladolid")
    laliga_df.loc[mask_away, "APoss"] = poss

# === 3. Loop semua barcelona possession (AGAINST) ===
for _, row in barca_against_df.iterrows():
    match_date = row["date"]
    poss = row["possession"]

    # Lawan Barcelona HOME → berarti possession against masuk ke HPoss
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Valladolid")
    laliga_df.loc[mask_home, "HPoss"] = poss

    # Lawan Real Valladolid AWAY → possession against masuk ke APoss
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Valladolid")
    laliga_df.loc[mask_away, "APoss"] = poss

# === 4. Simpan hasil ===
laliga_df.to_excel("laliga_updated_poss_valladolid.xlsx", index=False)
print("Data possession Valladolid berhasil dimasukkan ke HPoss/APoss.")
