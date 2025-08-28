import time
import base64
from collections import Counter
from io import BytesIO
import os
import json
import random
import requests
from PIL import Image

import undetected_chromedriver as uc  

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ================================== CONFIG ===========================================
imagedownloaded = 0
download_dir = "..." # Name of the image folder
# =====================================================================================

# Loads the JSON and chooses ONE query (r, n or i) by claim_id

def load_and_select_queries(json_filepath, 
                             p_relevant=0.6, 
                             p_neutral=0.3, 
                             p_irrelevant=0.1):
    assert abs(p_relevant + p_neutral + p_irrelevant - 1.0) < 1e-6
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selected_queries = {}
    print(f"Total items: {len(data)}")

    claim_ids = [item['claim_id'] for item in data]
    counts = Counter(claim_ids)
    duplicates = {cid: c for cid, c in counts.items() if c > 1}
    print("=================Checking For Duplicated Items===================")
    print(f"Unique claim_ids: {len(set(claim_ids))}")
    print(f"Duplicates: {duplicates}")

    for item in data:
        claim_id = item['claim_id']
        r = random.random()
        if r < p_relevant:
            selected_queries[claim_id] = {
                "claimID": claim_id,
                "query": item['query_relevant'],
                "imgCorr": "r"
            }
        elif r < p_relevant + p_neutral:
            selected_queries[claim_id] = {
                "claimID": claim_id,
                "query": item['query_neutral'],
                "imgCorr": "n"
            }
        else:
            selected_queries[claim_id] = {
                "claimID": claim_id,
                "query": item['query_irrelevant'],
                "imgCorr": "i"
            }

    return selected_queries

def safe_click(driver, element):
    try:
        element.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", element)
        
def image_exists_for_claim(download_dir, claim_id):
    if not os.path.exists(download_dir):
        return False
    prefix = f"{claim_id}_"
    for filename in os.listdir(download_dir):
        if filename.startswith(prefix):
            return True
    return False

# Downloads one image per query

def download_google_images(search_query: str, claimID: str, imgCorr: str):
    global imagedownloaded, tags
       
    if image_exists_for_claim(download_dir, claimID):
        print(f"Image already exists for {claimID}, skip.")
        return
    driver.get("https://images.google.com/")
    box = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//textarea[@name='q']"))
    )
    box.send_keys(search_query)
    box.send_keys(Keys.ENTER)
    time.sleep(1)

    try:
        containers = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.H8Rx8c"))
        )
        if not containers:
            print("No image containers found")
            return

        img = containers[0].find_element(By.CSS_SELECTOR, "g-img img")
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "g-img img")))
        safe_click(driver, img)
        time.sleep(1)

        big_image_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.p7sI2.PUxBg"))
        )
        imgs = big_image_container.find_elements(By.TAG_NAME, "img")
        src = imgs[0].get_attribute("src")

    except TimeoutException:
        print(f"No image found for query: {search_query}")
        return

    # Save the image
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, f'{claimID}_{imgCorr}.jpeg')

    if src.startswith('http'):
        try:
            result = requests.get(src, allow_redirects=True, timeout=10)
            if 'image' not in result.headers.get('Content-Type', ''):
                print(f"URL does not contain an image: {src}")
                return

            with open(file_path, 'wb') as f:
                f.write(result.content)

            img = Image.open(file_path).convert('RGB')
            img.save(file_path, 'JPEG')

            imagedownloaded += 1
            print(f'Image saved: {file_path} ({imagedownloaded}/{len(tags)})')
        except Exception as e:
            print(f"Failed to download image: {e}")
            try:
                os.unlink(file_path)
            except:
                pass

    elif src.startswith('data:image'):
        try:
            header, encoded = src.split(',', 1)
            img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
            img.save(file_path, 'JPEG')
            print(f'Image saved from Base64: {file_path}')
        except Exception as e:
            print(f"Failed to decode/save base64 image: {e}")


# MAIN SCRIPT 

if __name__ == "__main__":
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=options, headless=False)

    driver.get("https://images.google.com/")
    print("Waiting 5 seconds for you to accept cookies manually...")
    time.sleep(5)

    tags = load_and_select_queries('.json') # Dataset path

    for claimID, tag_data in tags.items():
        tag = tag_data['query']
        imgCorr = tag_data['imgCorr']
        print(f'{"="*10} Downloading for: {claimID} ({imgCorr}) {"="*10}')
        download_google_images(tag, claimID, imgCorr)
        print(f'{"="*10} Finished: {claimID} {"="*10}')

    driver.quit()