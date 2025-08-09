from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://fbref.com/en/comps/12/2024-2025/schedule/2024-2025-La-Liga-Scores-and-Fixtures"

def scrape_xg():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=1000)  # 1 detik delay tiap action
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })

        page.goto(URL, timeout=60000)
        page.wait_for_selector("table[id^='sched']")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        page.wait_for_timeout(3000)  # tunggu 3 detik setelah scroll

        html = page.inner_html("table[id^='sched']")
        soup = BeautifulSoup(html, "lxml")

        data = []
        tbody = soup.find("tbody")
        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []):
                continue

            date_cell = row.find(attrs={"data-stat": "date"})
            home_cell = row.find(attrs={"data-stat": "home_team"})
            away_cell = row.find(attrs={"data-stat": "away_team"})
            hxg_cell = row.find(attrs={"data-stat": "home_xg"})
            axg_cell = row.find(attrs={"data-stat": "away_xg"})

            if not all([date_cell, home_cell, away_cell, hxg_cell, axg_cell]):
                continue

            date = date_cell.get_text(strip=True)
            home_team = home_cell.get_text(strip=True)
            away_team = away_cell.get_text(strip=True)
            hxg = hxg_cell.get_text(strip=True)
            axg = axg_cell.get_text(strip=True)

            data.append({
                "date": date,
                "home_team": home_team,
                "hxg": float(hxg) if hxg else None,
                "axg": float(axg) if axg else None,
                "away_team": away_team
            })

        browser.close()

    df = pd.DataFrame(data)
    df.to_csv("laliga_xg.csv", index=False)
    print(df.head())
    print(f"Total {len(df)} pertandingan disimpan ke premier_league_xg_tes.csv")

if __name__ == "__main__":
    scrape_xg()
