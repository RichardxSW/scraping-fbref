from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment

URL = "https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures"

def debug_comments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=60000)

        html = page.content()
        soup = BeautifulSoup(html, "lxml")
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))

        print(f"Total komentar ditemukan: {len(comments)}\n")
        for i, c in enumerate(comments[:10]):  # print 10 komentar pertama
            print(f"--- Komentar #{i+1} ---")
            print(c[:1000])  # print 1000 karakter pertama saja
            print("\n\n")

        browser.close()

if __name__ == "__main__":
    debug_comments()
