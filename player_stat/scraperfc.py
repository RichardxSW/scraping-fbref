from pathlib import Path
import ScraperFC as sfc
import pandas as pd
import soccerdata as sd

# fb=sfc.FBref()

sddata=sd.FBref(
    leagues=['ENG-Premier League'],
    seasons=["2425"],
    # proxy='tor',
    # no_cache=True,
    # no_store=True
    # data_dir=Path("liverpool_")
)

# result = fb.scrape_all_stats(
#     year="2024-2025",
#     league="EPL"
# )

# print(result)

# std = sddata.read_player_season_stats(stat_type="standard")
# keeper = sddata.read_player_season_stats(stat_type="keeper")
# shot = sddata.read_player_season_stats(stat_type="shooting")
# passing = sddata.read_player_season_stats(stat_type="passing")
# defense = sddata.read_player_season_stats(stat_type="defense")
# possession = sddata.read_player_season_stats(stat_type="possession")
# misc = sddata.read_player_season_stats(stat_type="misc")

team = sddata.read_team_match_stats(stat_type='passing', team='Liverpool')
print(team.head())
# --- Ekspor ke Excel: satu file, multi-sheet ---
# OUTFILE = "premier.xlsx"
# with pd.ExcelWriter(OUTFILE, engine="xlsxwriter") as xls:
#     for name, df in {
#         "standard": std,
#         "keeper": keeper,
#         "shooting": shot,
#         "passing":  passing,
#         "defense": defense, 
#         "possession": possession, 
#         "misc": misc
#         # tambah sheet lain di sini
#     }.items():
#         if df is not None and not df.empty:
#             df.to_excel(xls, sheet_name=name[:31], index=True)

# print(result.head())