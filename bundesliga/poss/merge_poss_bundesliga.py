from pathlib import Path
import pandas as pd

folder = Path("laliga")
# === 1. Baca file ===
laliga_df = pd.read_excel("bundesliga/poss/bundesliga_updated_poss_Heidenheim.xlsx")
barca_for_df = pd.read_csv("bundesliga/poss/bundesliga_possession_tes.csv")
barca_against_df = pd.read_csv("bundesliga/poss/bundesliga_possession_against.csv")

# Pastikan kolom date jadi datetime
laliga_df["date"] = pd.to_datetime(laliga_df["date"], errors="coerce", dayfirst=True)
barca_for_df["date"] = pd.to_datetime(barca_for_df["date"], format="%Y-%m-%d", errors="coerce")
barca_against_df["date"] = pd.to_datetime(barca_against_df["date"], format="%Y-%m-%d", errors="coerce")

# === 2. Loop semua Nottingham Forest possession (FOR) ===
for _, row in barca_for_df.iterrows():
    match_date = row["date"]
    poss = row["possession"]

    # Nottingham Forest sebagai HOME
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Holstein Kiel")
    laliga_df.loc[mask_home, "HPoss"] = poss

    # Real Nottingham Forest sebagai AWAY
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Holstein Kiel")
    laliga_df.loc[mask_away, "APoss"] = poss

# === 3. Loop semua Nottingham Forest possession (AGAINST) ===
for _, row in barca_against_df.iterrows():
    match_date = row["date"]
    poss = row["possession"]

    # Lawan Nottingham Forest HOME → berarti possession against masuk ke HPoss
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Holstein Kiel")
    laliga_df.loc[mask_home, "HPoss"] = poss

    # Lawan Real Nottingham Forest AWAY → possession against masuk ke APoss
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Holstein Kiel")
    laliga_df.loc[mask_away, "APoss"] = poss

# === 4. Simpan hasil ===
laliga_df.to_excel("bundesliga/poss/bundesliga_updated_poss_Holstein_Kiel.xlsx", index=False)
print("Data possession Holstein Kiel berhasil dimasukkan ke HPoss/APoss.")
