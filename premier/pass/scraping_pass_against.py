from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
import pandas as pd
import time

URL = "https://fbref.com/en/squads/b74092de/2024-2025/matchlogs/c9/passing/Ipswich-Town-Match-Logs-Premier-League"

def scrape_against_possession():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

        page.goto(URL, timeout=10000)

        # Klik tab "Against Ipswich Town"
        tab = page.locator("a.sr_preset", has_text="Against Ipswich Town")
        if tab.count() == 0:
            tab = page.locator("text=Against Ipswich Town")

        if tab.count() == 0:
            print("Tab 'Against Ipswich Town' tidak ditemukan.")
            page.screenshot(path="debug_no_tab.png")
            open("debug_no_tab.html", "w", encoding="utf-8").write(page.content())
            browser.close()
            return

        tab.scroll_into_view_if_needed()
        tab.click(force=True)
        print("Tab 'Against Ipswich Town' diklik")
        time.sleep(0.5)

        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except TimeoutError:
            pass

        table_selector = "table[id^='matchlogs_against']"
        try:
            page.wait_for_selector(f"{table_selector} tbody tr", timeout=10000)

            print("Data tabel ditemukan")
        except TimeoutError:
            print("Tidak menemukan rows di matchlogs_against, mencoba fallback ke table[id^='matchlogs']")
            found = False
            tables = page.query_selector_all("table[id^='matchlogs']")
            for t in tables:
                tid = t.get_attribute("id")
                rows = t.query_selector_all("tbody tr")
                if len(rows) > 0:
                    table_selector = f"table#{tid}"
                    found = True
                    print(f"Fallback: menggunakan table#{tid} ({len(rows)} rows)")
                    break

            if not found:
                print("Tidak menemukan tabel berisi baris.")
                page.screenshot(path="debug_against_no_rows.png")
                open("debug_against_no_rows.html", "w", encoding="utf-8").write(page.content())
                browser.close()
                return

        html = page.inner_html(table_selector)
        soup = BeautifulSoup(html, "lxml")

        data = []
        tbody = soup.find("tbody")
        if not tbody:
            print("tbody tidak ditemukan di HTML tabel.")
            open("debug_against_table.html", "w", encoding="utf-8").write(html)
            browser.close()
            return

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class"):
                continue

            date_cell = row.find(attrs={"data-stat": "date"})
            opponent_cell = row.find(attrs={"data-stat": "opponent"})
            pass_cell = row.find(attrs={"data-stat": "passes"})
            success_pass_cell = row.find(attrs={"data-stat": "passes_completed"})
            # venue_cell = row.find(attrs={"data-stat": "venue"})
            # possession_cell = row.find(attrs={"data-stat": "possession"})
            # score_cell = row.find(attrs={"data-stat": "score"})

            date = date_cell.get_text(strip=True) if date_cell else None
            opponent = opponent_cell.get_text(strip=True) if opponent_cell else None
            passes = pass_cell.get_text(strip=True) if pass_cell else None
            success_pass = success_pass_cell.get_text(strip=True) if success_pass_cell else None
            # venue = venue_cell.get_text(strip=True) if venue_cell else None
            # possession = possession_cell.get_text(strip=True).replace("%", "") if possession_cell else None
            # score = score_cell.get_text(strip=True) if score_cell else None

            data.append({
                "date": date,
                "opponent": opponent,
                "passes": passes,
                "success_passes": success_pass,
                # "venue": venue,
                # "possession": float(possession) if possession else None,
                # "score": score,
            })

        browser.close()

    df = pd.DataFrame(data)
    df.to_csv("premier/pass/liverpool_pass_against.csv", index=False)
    print(df.head())
    print(f"Total {len(df)} baris disimpan ke liverpool_pass_against.csv")

if __name__ == "__main__":
    scrape_against_possession()
