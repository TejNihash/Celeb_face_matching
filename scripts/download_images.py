import os
from duckduckgo_search import DDGS
from PIL import Image
import requests
from io import BytesIO

# Config
CELEB_LIST_FILE = 'celebs.txt'
SAVE_DIR = 'data'
IMAGES_PER_CELEB = 5

def load_celeb_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def download_images_for_celeb(name, save_folder, required_images=5, search_limit=15):
    print(f"[INFO] Searching for images of {name}")
    celeb_dir = os.path.join(save_folder, name.replace(" ", "_"))
    os.makedirs(celeb_dir, exist_ok=True)

    downloaded = 0
    with DDGS() as ddgs:
        results = ddgs.images(name, max_results=search_limit)
        for idx, result in enumerate(results):
            if downloaded >= required_images:
                break
            try:
                url = result["image"]
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img.save(os.path.join(celeb_dir, f"{downloaded}.jpg"))
                print(f"[✓] Saved {name} - {downloaded}.jpg")
                downloaded += 1
            except Exception as e:
                print(f"[X] Skipping bad image for {name}: {e}")

    if downloaded < required_images:
        print(f"[!] Only found {downloaded}/{required_images} valid images for {name}")


def main():
    celeb_names = load_celeb_names(CELEB_LIST_FILE)
    for celeb in celeb_names:
        download_images_for_celeb(celeb, SAVE_DIR, IMAGES_PER_CELEB)

if __name__ == "__main__":
    main()
