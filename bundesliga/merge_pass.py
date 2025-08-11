from pathlib import Path
import pandas as pd

folder = Path("laliga")
# === 1. Baca file ===
laliga_df = pd.read_excel("bundesliga_updated_poss.xlsx")
barca_for_df = pd.read_csv("bundesliga/pass/bundesliga_pass.csv")
barca_against_df = pd.read_csv("bundesliga/pass/bundesliga_pass_against.csv")

# Pastikan kolom date jadi datetime
laliga_df["date"] = pd.to_datetime(laliga_df["date"], errors="coerce")
barca_for_df["date"] = pd.to_datetime(barca_for_df["date"], errors="coerce")
barca_against_df["date"] = pd.to_datetime(barca_against_df["date"], errors="coerce")

# === 2. Loop semua Bayern Munich possession (FOR) ===
for _, row in barca_for_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Bayern Munich sebagai HOME
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Bayern Munich")
    laliga_df.loc[mask_home, "HTP"] = passes
    laliga_df.loc[mask_home, "HSP"] = success_passes

    # Real Bayern Munich sebagai AWAY
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Bayern Munich")
    laliga_df.loc[mask_away, "ATP"] = passes
    laliga_df.loc[mask_away, "ASP"] = success_passes

# === 3. Loop semua Bayern Munich possession (AGAINST) ===
for _, row in barca_against_df.iterrows():
    match_date = row["date"]
    passes = row["passes"]
    success_passes = row["success_passes"]

    # Lawan Bayern Munich HOME → berarti possession against masuk ke HPoss
    mask_home = (laliga_df["date"] == match_date) & (laliga_df["AwayTeam"] == "Bayern Munich")
    laliga_df.loc[mask_home, "HTP"] = passes
    laliga_df.loc[mask_home, "HSP"] = success_passes

    # Lawan Real Bayern Munich AWAY → possession against masuk ke APoss
    mask_away = (laliga_df["date"] == match_date) & (laliga_df["HomeTeam"] == "Bayern Munich")
    laliga_df.loc[mask_away, "ATP"] = passes
    laliga_df.loc[mask_away, "ASP"] = success_passes
 
# === 4. Simpan hasil ===
laliga_df.to_excel("bundesliga/pass/bundesliga_updated_pass_Bayern_Munich.xlsx", index=False)
print("Data pass Bayern Munich berhasil dimasukkan.")
