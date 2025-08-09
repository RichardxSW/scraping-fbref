from playwright.sync_api import sync_playwright
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures"

def scrape_xg():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=60000)

        # Cari ID tabel yang mengandung jadwal (diawali 'sched_')
        table_selector = "table[id^='sched']"
        page.wait_for_selector(table_selector, timeout=60000)

        html = page.inner_html(table_selector)
        soup = BeautifulSoup(html, "lxml")

        data = []
        for row in soup.find("tbody").find_all("tr"):
            if "thead" in row.get("class", []):
                continue

            date = row.find(attrs={"data-stat": "date"}).get_text(strip=True)
            home_team = row.find(attrs={"data-stat": "home_team"}).get_text(strip=True)
            away_team = row.find(attrs={"data-stat": "away_team"}).get_text(strip=True)
            hxg = row.find(attrs={"data-stat": "home_xg"}).get_text(strip=True)
            axg = row.find(attrs={"data-stat": "away_xg"}).get_text(strip=True)

            data.append({
                "date": date,
                "home_team": home_team,
                "hxg": float(hxg) if hxg else None,
                "axg": float(axg) if axg else None,
                "away_team": away_team
            })

        browser.close()

    df = pd.DataFrame(data)
    df.to_csv("premier_league_xg_tes.csv", index=False)
    print(df.head())
    print(f"Total {len(df)} pertandingan disimpan ke premier_league_xg.csv")

if __name__ == "__main__":
    scrape_xg()
