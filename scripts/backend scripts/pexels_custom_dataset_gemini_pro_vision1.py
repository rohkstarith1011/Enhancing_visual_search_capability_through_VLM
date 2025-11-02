import os
import requests
from PIL import Image
import io
import json
from tqdm import tqdm
import time

# ==============================================================================
# 1. Configuration
# ==============================================================================
# --- ⚠️ EACH TEAMMATE: PASTE YOUR *OWN* NEW, SECRET UNSPLASH KEY HERE ---
UNSPLASH_ACCESS_KEY = "5py_LGQOU_t0bN3J6WPblqBZwO3MLLQ6VzxLzRTjrgw"

# --- ⚠️ EACH TEAMMATE: CHANGE THIS TO YOUR ASSIGNED QUERY ---
SEARCH_QUERY = "Chinese foods"

# --- Number of images you want to get for THIS query ---
IMAGES_TO_DOWNLOAD = 400 # 400 is a safe number (~13 API requests)

# --- Base project directory ---
project_dir = "/content/drive/My Drive/final year project"

# --- DYNAMIC OUTPUT FOLDER (This is the key change) ---
# Creates a safe folder name like "dataset_indian_sweets"
project_dir = "/content/drive/My Drive/final year project" 
output_dir = os.path.join(project_dir, "Culinary_Caption_Dataset_Demo")
image_output_dir = os.path.join(output_dir, "images")
os.makedirs(image_output_dir, exist_ok=True)
output_json_path = os.path.join(output_dir, "metadata.jsonl")

# ==============================================================================
# 2. Fetch Image Metadata from Unsplash
# ==============================================================================
if UNSPLASH_ACCESS_KEY == "YOUR_NEW_UNSPLASH_KEY_HERE":
    print("❌ ERROR: Please paste your new Unsplash API key into the script.")
else:
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    unsplash_url = "https://api.unsplash.com/search/photos"

    all_photos = []
    seen_photo_ids = set()
    
    print(f"--- Starting query: '{SEARCH_QUERY}' ---")
    print(f"Goal: Collect {IMAGES_TO_DOWNLOAD} images.")
    print("⚠️ This will be SLOW to respect the 50 requests/hour Demo limit.")
    
    page_num = 1
    
    # Keep looping until we hit our download goal
    while len(all_photos) < IMAGES_TO_DOWNLOAD:
        query_params = {
            "query": SEARCH_QUERY,
            "per_page": 30, # Max allowed by Unsplash is 30
            "page": page_num
        }
        
        try:
            # --- THIS IS THE API REQUEST ---
            response = requests.get(unsplash_url, headers=headers, params=query_params)
            response.raise_for_status() # This will error if you are rate-limited (403)
            
            page_results = response.json().get("results", [])

            if not page_results:
                print(f"\nNo more results for query '{SEARCH_QUERY}'. Stopping.")
                break 

            new_photos_found = 0
            for photo in page_results:
                if photo['id'] not in seen_photo_ids:
                    seen_photo_ids.add(photo['id'])
                    all_photos.append(photo)
                    new_photos_found += 1
                    if len(all_photos) >= IMAGES_TO_DOWNLOAD:
                        break

            print(f"\nCollected {len(all_photos)} / {IMAGES_TO_DOWNLOAD} total images.")
            
            if len(all_photos) >= IMAGES_TO_DOWNLOAD:
                print("Goal reached. Stopping collection.")
                break

            page_num += 1
            
            # --- CRITICAL RATE LIMIT DELAY ---
            # Wait ~73 seconds to stay under the 50 requests/hour limit.
            print(f"Request successful. Waiting 73 seconds before next request...")
            time.sleep(73) 
        
        except requests.exceptions.RequestException as e:
            print(f"\nAPI request failed: {e}.")
            print("This is likely a 403 Forbidden (rate limit) error.")
            print("Waiting 1 hour before retrying...")
            time.sleep(3601) # Wait for the hour to reset
            continue # Retry this same request

    photos_to_process = all_photos[:IMAGES_TO_DOWNLOAD]
    print(f"\nSuccessfully fetched metadata for {len(photos_to_process)} total images.")

# ==============================================================================
# 3. Download Images and Create "To-Do" File
# ==============================================================================
print(f"Downloading images and creating metadata 'to-do' list at: {output_json_path}")

# We use "w" (write mode) here because this is a new, unique file for this query
with open(output_json_path, "w") as f:
    for photo in tqdm(photos_to_process, desc="Downloading & Saving Images"):
        try:
            image_url = photo['urls']['regular'] # Use 'regular' for good quality
            image_id = photo['id']
            download_notify_url = photo['links']['download_location'] # Get tracking URL

            image_response = requests.get(image_url, timeout=15)
            image_response.raise_for_status()
            
            raw_image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            image_filename = f"{image_id}.jpg"
            image_save_path = os.path.join(image_output_dir, image_filename)
            raw_image.save(image_save_path, "JPEG", quality=90)
            
            relative_image_path = os.path.join("images", image_filename)
            # We save an empty caption as a placeholder
            record = {"image_path": relative_image_path, "caption": ""} 
            
            f.write(json.dumps(record) + "\n")
            
            # Notify Unsplash of the download (REQUIRED by their terms)
            requests.get(download_notify_url, headers=headers)
            
        except Exception as e:
            print(f"\nSkipping image {photo.get('id')} due to download error: {e}")

print(f"\n✅ Stage 1 Complete. All images downloaded to: {image_output_dir}")
print(f"➡️ Now you can run the '02_caption_generation.py' script on this folder.")