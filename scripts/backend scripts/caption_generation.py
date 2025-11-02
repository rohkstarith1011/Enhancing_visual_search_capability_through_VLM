import os
import json
from PIL import Image
from tqdm import tqdm
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, BlockedPromptException
import shutil # Needed for copying and removing files/directories

# ==============================================================================
# 1. Configuration
# ==============================================================================
YOUR_GEMINI_API_KEY = "AIzaSyC_xerMSw1HbtyxHOj6xmR5P-e8M6PK5UE" # ❗ Ensure this is filled

# --- Configure the Gemini client ---
# ... (Configuration code remains the same) ...
if YOUR_GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE" or not YOUR_GEMINI_API_KEY:
    print("❌ ERROR: Please paste your Gemini API key.")
    exit()
else:
    try:
        genai.configure(api_key=YOUR_GEMINI_API_KEY)
        print("✅ Gemini API key configured successfully.")
    except Exception as e:
        print(f"An error occurred during Gemini configuration: {e}")
        exit()

# --- File Paths ---
project_dir = "/content/drive/My Drive/final year project"
output_dir = os.path.join(project_dir, "Culinary_Caption_Dataset_Demo")
input_json_path = os.path.join(output_dir, "metadata.jsonl")
output_json_path = os.path.join(output_dir, "metadata_captioned.jsonl")
drive_image_base_dir = os.path.join(output_dir, "images") # Base dir on Drive

# --- Local Colab Storage Paths ---
local_temp_dir = "/content/temp_images" # Temporary local directory

# --- Batching Configuration ---
BATCH_SIZE = 100 # Process 100 images at a time locally

# ==============================================================================
# 2. Setup the Captioning Model (Unchanged)
# ==============================================================================
print("Setting up Gemini model...")
model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Gemini 2.5 Flash model is ready.")

# ==============================================================================
# 3. Define Captioning Function (Unchanged - includes safety settings)
# ==============================================================================
def generate_gemini_caption_single_call(model, image, retries=3):
    # ... (Your captioning function remains exactly the same) ...
    prompt = (
        "You are an expert chef and food blogger. First, identify the specific name of the culinary dish in the image. "
        "Second, write a detailed, vocabulary-rich sentence describing that dish. "
        "Highlight its key ingredients, colors, textures, and presentation, as if for a high-end food blog. "
        "Format your response as:\n"
        "Dish Name: [Identified Name]\n"
        "Description: [Descriptive Sentence]"
    )
    custom_safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    for attempt in range(retries):
        try:
            response = model.generate_content(
                [prompt, image],
                safety_settings=custom_safety_settings
            )
            response_text = response.text.strip()
            dish_name = "Unknown Dish"
            description = "A photo of a food dish."
            lines = response_text.split('\n')
            if len(lines) >= 2:
                if lines[0].startswith("Dish Name:"):
                    dish_name = lines[0].replace("Dish Name:", "").strip()
                if lines[1].startswith("Description:"):
                    description = lines[1].replace("Description:", "").strip()
            elif len(lines) == 1 and not ("I'm not able" in response_text or "I cannot" in response_text):
                description = response_text
            if len(description) < 20 or "unknown" in dish_name.lower():
                return f"A high-quality photo of {dish_name}." if dish_name != "Unknown Dish" else "A photo of a food dish."
            return description
        except BlockedPromptException as e:
            print(f"\n  - Prompt STILL blocked by safety settings. Skipping.")
            return None
        except ValueError as e:
            if "response was blocked" in str(e).lower():
                 print(f"\n  - Prompt STILL blocked (ValueError). Skipping.")
                 return None
            else:
                 print(f"\n  - Value Error on attempt {attempt + 1}: {e}. Retrying...")
                 time.sleep(5 + attempt * 5)
        except Exception as e:
            print(f"\n  - API Error on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(5 + attempt * 5)
    print(f"\n  - Skipping image after multiple API failures.")
    return None

# ==============================================================================
# 4. Load "To-Do" List and "Done" List (Unchanged)
# ==============================================================================
# ... (This section remains exactly the same, determining 'records_to_process') ...
try:
    with open(input_json_path, "r") as f:
        all_records = [json.loads(line) for line in f]
    print(f"Loaded {len(all_records)} total records from {input_json_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Input file not found: {input_json_path}")
    all_records = []
    exit()

processed_image_paths = set()
if os.path.exists(output_json_path):
    try:
        with open(output_json_path, "r") as f:
            for line in f:
                processed_image_paths.add(json.loads(line)['image_path'])
        print(f"Found {len(processed_image_paths)} already-captioned images in {output_json_path}.")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {output_json_path}. Starting fresh. Error: {e}")

records_to_process = [r for r in all_records if r['image_path'] not in processed_image_paths]
print(f"➡️ Starting captioning for {len(records_to_process)} remaining images...")

# ==============================================================================
# 5. Process Images in Batches (***NEW LOGIC***)
# ==============================================================================
processed_count_this_run = 0
daily_limit = 10000

# Loop through the records_to_process in chunks (batches)
for i in tqdm(range(0, min(len(records_to_process), daily_limit), BATCH_SIZE), desc="Overall Progress"):
    
    batch_records = records_to_process[i:min(i + BATCH_SIZE, daily_limit)]
    batch_results = [] # Store results for this batch in memory

    # --- 1. Prepare Local Batch Directory ---
    if os.path.exists(local_temp_dir):
        shutil.rmtree(local_temp_dir) # Clean up previous batch
    os.makedirs(local_temp_dir)

    # --- 2. Copy Images for Batch from Drive to Local ---
    print(f"\nCopying batch {i//BATCH_SIZE + 1} ({len(batch_records)} images) to local storage...")
    for record in tqdm(batch_records, desc="Copying to local", leave=False):
        drive_image_path = os.path.join(output_dir, record['image_path'])
        local_image_path = os.path.join(local_temp_dir, os.path.basename(record['image_path']))
        try:
            if os.path.exists(drive_image_path):
                shutil.copy2(drive_image_path, local_image_path)
            else:
                 print(f"\nWarning: Source image not found on Drive: {drive_image_path}")
        except Exception as copy_e:
            print(f"\nError copying {drive_image_path} to local: {copy_e}")
            
    # --- 3. Process Batch Locally ---
    print(f"Processing batch {i//BATCH_SIZE + 1}...")
    for record in tqdm(batch_records, desc="Captioning Batch", leave=False):
        local_image_path = os.path.join(local_temp_dir, os.path.basename(record['image_path']))
        
        if not os.path.exists(local_image_path):
            print(f"\nSkipping {record['image_path']}: Not found in local batch (copy failed?).")
            continue # Skip if copy failed

        try:
            raw_image = Image.open(local_image_path)
            caption = generate_gemini_caption_single_call(model, raw_image)

            if caption is None: # Skip if captioning failed or blocked
                continue

            # Add successful result to in-memory list
            record['caption'] = caption
            batch_results.append(record)

            # Rate Limiting Delay (Reduced - adjust if needed)
            time.sleep(0.05) # Keep the faster delay for now

        except FileNotFoundError:
            print(f"\nSkipping {record['image_path']}: File not found during Image.open() from local.")
        except Exception as e:
            print(f"\nSkipping {record['image_path']} due to critical error during processing: {e}")

    # --- 4. Append Batch Results to Drive File ---
    if batch_results:
        print(f"Appending {len(batch_results)} results for batch {i//BATCH_SIZE + 1} to {output_json_path}...")
        try:
            with open(output_json_path, "a") as f:
                for result_record in batch_results:
                    f.write(json.dumps(result_record) + "\n")
                f.flush() # Flush once after writing the whole batch
            processed_count_this_run += len(batch_results)
        except Exception as write_e:
             print(f"\nError writing batch results to Google Drive: {write_e}")

    # --- 5. Clean up Local Directory ---
    print(f"Cleaning up local files for batch {i//BATCH_SIZE + 1}...")
    shutil.rmtree(local_temp_dir)

# ==============================================================================
# 6. Final Status Message (Adjusted for batching)
# ==============================================================================
print(f"\n✅ Daily batch processing finished. Processed approximately {processed_count_this_run} images in this run.")
print(f"Total captioned images saved in: {output_json_path}")

# Estimate remaining based on initial list and total processed so far
total_processed_so_far = len(processed_image_paths) + processed_count_this_run
total_remaining_estimated = len(all_records) - total_processed_so_far

if total_remaining_estimated <= 0:
     print("🎉 All images appear to have been successfully captioned! Your project is complete.")
elif processed_count_this_run >= (daily_limit - BATCH_SIZE): # If we processed close to the daily limit
     print(f"🔔 You've likely hit your daily limit! Approximately {total_remaining_estimated} images might be left.")
     print("Please wait 24 hours for your API quota to reset, then run this script again.")
else:
     print(f"✅ Script finished processing the current session. Approximately {total_remaining_estimated} images might be left due to skips or prior interruptions.")
     print("You can run the script again to continue.")