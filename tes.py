import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. Data historis dummy
# ===============================
# Fitur: [MMR difference, role balance score, avg hero pool size difference]
# Label: 1 = Team A menang, 0 = kalah
X_train = [
    [50, 0, 1],   # MMR lebih tinggi, role balance bagus, hero pool lebih luas
    [-100, 1, -1],
    [200, 0, 0],
    [0, 0, 0],
    [-50, 1, 1],
    [80, 0, -1]
]
y_train = [1, 0, 1, 0, 0, 1]

# Latih model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 2. Fungsi ekstraksi fitur
# ===============================
def extract_features(teamA, teamB, player_data):
    # Ambil mmr & role tiap tim
    mmrA = [p["mmr"] for p in player_data if p["player_id"] in teamA]
    mmrB = [p["mmr"] for p in player_data if p["player_id"] in teamB]

    rolesA = [p["role"] for p in player_data if p["player_id"] in teamA]
    rolesB = [p["role"] for p in player_data if p["player_id"] in teamB]

    poolA = [p["hero_pool"] for p in player_data if p["player_id"] in teamA]
    poolB = [p["hero_pool"] for p in player_data if p["player_id"] in teamB]

    # Fitur sederhana
    mmr_diff = np.mean(mmrA) - np.mean(mmrB)   # selisih rata-rata MMR
    role_balance = abs(len(set(rolesA)) - len(set(rolesB)))  # selisih variasi role
    hero_pool_diff = np.mean(poolA) - np.mean(poolB)  # rata-rata hero pool

    return [mmr_diff, role_balance, hero_pool_diff]

# ===============================
# 3. Data antrean 10 pemain
# ===============================
queue = [
    {"player_id": 1, "mmr": 1800, "role": "tank",    "hero_pool": 15},
    {"player_id": 2, "mmr": 1750, "role": "jungle",  "hero_pool": 20},
    {"player_id": 3, "mmr": 1900, "role": "mid",     "hero_pool": 12},
    {"player_id": 4, "mmr": 1700, "role": "gold",    "hero_pool": 10},
    {"player_id": 5, "mmr": 1650, "role": "exp",     "hero_pool": 8},
    {"player_id": 6, "mmr": 1850, "role": "tank",    "hero_pool": 18},
    {"player_id": 7, "mmr": 1720, "role": "jungle",  "hero_pool": 14},
    {"player_id": 8, "mmr": 1780, "role": "mid",     "hero_pool": 11},
    {"player_id": 9, "mmr": 1600, "role": "gold",    "hero_pool": 9},
    {"player_id": 10,"mmr": 1680, "role": "exp",     "hero_pool": 7},
]

# ===============================
# 4. Cari kombinasi terbaik
# ===============================
players = [p["player_id"] for p in queue]
best_match = None
best_diff = 1.0

for teamA in itertools.combinations(players, 5):
    teamB = [p for p in players if p not in teamA]

    features = extract_features(teamA, teamB, queue)
    prob_win = model.predict_proba([features])[0][1]  # peluang Tim A menang
    diff = abs(prob_win - 0.5)

    if diff < best_diff:
        best_diff = diff
        best_match = (teamA, teamB, prob_win)

print("Best Match:")
print("Tim A:", best_match[0])
print("Tim B:", best_match[1])
print("Peluang Tim A menang:", round(best_match[2], 3))
