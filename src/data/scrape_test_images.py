"""
scrape_test_images.py
======================
Scrapes extra face photos per player from Bing Images as a *supplementary*
test set — useful for evaluating the model on images that never touched
the Kaggle training distribution at all (different photographers, poses,
lighting), which is a stronger generalization check than a held-out split
of the same source dataset.

Requires Google Chrome installed. Saves into data/raw/test/<player>/.
"""

import hashlib
import time
from io import BytesIO

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from src.config import TEST_SOURCE_DIR, TRAIN_SOURCE_DIR

IMAGES_PER_PLAYER = 20


def get_player_names(base_dir=TRAIN_SOURCE_DIR) -> list[str]:
    return [f.name for f in base_dir.iterdir() if f.is_dir()]


def setup_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--headless=new")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def save_image(folder, url: str) -> None:
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        name = hashlib.sha1(response.content).hexdigest()[:10]
        img.save(folder / f"{name}.jpg", "JPEG", quality=90)
    except Exception:
        pass


def scrape_player(driver: webdriver.Chrome, player: str, images_per_player: int = IMAGES_PER_PLAYER) -> None:
    print(f"Scraping: {player}")

    folder = TEST_SOURCE_DIR / player
    folder.mkdir(parents=True, exist_ok=True)

    player_name = player.replace("_", " ")
    queries = [
        f"{player_name} face close up",
        f"{player_name} portrait photo",
        f"{player_name} footballer face",
    ]

    urls: set[str] = set()
    for query in queries:
        if len(urls) >= images_per_player:
            break

        driver.get(f"https://www.bing.com/images/search?q={query.replace(' ', '+')}")
        time.sleep(2)

        for img in driver.find_elements(By.CSS_SELECTOR, "img.mimg"):
            src = img.get_attribute("src")
            if src and src.startswith("http"):
                urls.add(src)
            if len(urls) >= images_per_player:
                break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    for url in list(urls)[:images_per_player]:
        save_image(folder, url)

    print(f"  Saved: {len(list(folder.iterdir()))} images")


def scrape_all_players(images_per_player: int = IMAGES_PER_PLAYER) -> None:
    players = get_player_names()
    print(f"Found {len(players)} players: {players}\n")

    driver = setup_driver()
    try:
        for player in players:
            scrape_player(driver, player, images_per_player)
    finally:
        driver.quit()

    print("\nDone!")


if __name__ == "__main__":
    scrape_all_players()
