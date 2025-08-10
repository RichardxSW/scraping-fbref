from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://fbref.com/en/squads/2ac661d9/2024-2025/matchlogs/c20/possession/Holstein-Kiel-Match-Logs-Bundesliga"

def scrape_team_possession():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })

        page.goto(URL, timeout=10000)
        table_selector = "table[id^='matchlogs']"
        page.wait_for_selector(table_selector)

        html = page.inner_html(table_selector)
        soup = BeautifulSoup(html, "lxml")

        data = []
        tbody = soup.find("tbody")
        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []):
                continue

            # Tanggal
            date_cell = row.find(attrs={"data-stat": "date"})
            date = date_cell.get_text(strip=True) if date_cell else None

            # Lawan
            opponent_cell = row.find(attrs={"data-stat": "opponent"})
            opponent = opponent_cell.get_text(strip=True) if opponent_cell else None

            # Home/Away
            # venue_cell = row.find(attrs={"data-stat": "venue"})
            # venue = venue_cell.get_text(strip=True) if venue_cell else None

            # Possession
            possession_cell = row.find(attrs={"data-stat": "possession"})
            possession = possession_cell.get_text(strip=True).replace("%", "") if possession_cell else None

            # Score
            # score_cell = row.find(attrs={"data-stat": "score"})
            # score = score_cell.get_text(strip=True) if score_cell else None

            data.append({
                "date": date,
                "opponent": opponent,
                "possession": float(possession) if possession else None,
                # "venue": venue,
                # "score": score,
            })

        browser.close()

    df = pd.DataFrame(data)
    df.to_csv("bundesliga/poss/bundesliga_possession_tes.csv", index=False)
    print(df.head())
    print(f"Total {len(df)} baris data possession per match disimpan ke bundesliga/poss/bundesliga_possession.csv")

if __name__ == "__main__":
    scrape_team_possession()
