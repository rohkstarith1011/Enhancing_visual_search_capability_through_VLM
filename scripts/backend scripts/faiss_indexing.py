# Import necessary libraries
import torch # PyTorch for tensor operations and GPU usage
from transformers import CLIPProcessor, CLIPModel # Hugging Face library for CLIP model and processor
from torch.utils.data import DataLoader, Dataset # PyTorch utilities for handling datasets
from torchvision import transforms # For image preprocessing (ToTensor, Normalize)
from PIL import Image # Python Imaging Library for opening images
import numpy as np # NumPy for numerical operations, especially with embeddings
import faiss # Facebook AI Similarity Search library for vector indexing
import os # For interacting with the operating system (file paths, directories)
import json # For reading the JSON Lines metadata file
from tqdm import tqdm # For displaying progress bars during loops

# ==============================================================================
# 1. Configuration - **VERIFY ALL PATHS**
# ==============================================================================
# Define base directory for the project on Google Drive
project_dir = "/content/drive/My Drive/final year project"
# Define the specific dataset directory within the project
base_dataset_dir = os.path.join(project_dir, "Culinary_Caption_Dataset_Demo")

# --- Model to use for generating embeddings ---
# Path to the directory containing the fine-tuned CLIP model files
MODEL_DIR = os.path.join(project_dir, "clip_finetuned_culinary_cleaned_12345_then_6") # Your final best model path

# --- Input Cleaned Metadata ---
# Path to the JSON Lines file containing cleaned image paths and captions
METADATA_PATH = os.path.join(base_dataset_dir, "metadata_cleaned.jsonl")

# --- Base directory containing the actual image files ---
# Path to the folder where all the actual JPG/PNG image files are stored
IMAGE_BASE_DIR = os.path.join(base_dataset_dir, "images")

# --- Output Directory for Embeddings and Indexes ---
# Path to the folder where the generated embeddings and FAISS indexes will be saved
OUTPUT_DIR = os.path.join(project_dir, "faiss_indexes_cleaned")
# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Output Filenames ---
# Define the full path and filename for the image embeddings NumPy file
IMAGE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings_cleaned.npy")
# Define the full path and filename for the text embeddings NumPy file
TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "text_embeddings_cleaned.npy")
# Define the full path and filename for the FAISS index file for images
IMAGE_INDEX_FILE = os.path.join(OUTPUT_DIR, "images_cleaned.index")
# Define the full path and filename for the FAISS index file for text captions
TEXT_INDEX_FILE = os.path.join(OUTPUT_DIR, "texts_cleaned.index")
# Define the full path and filename for a JSON Lines file that stores metadata in the same order as the embeddings/indexes
METADATA_ORDER_FILE = os.path.join(OUTPUT_DIR, "metadata_order_cleaned.jsonl") # Save metadata order matching indexes

# --- Processing Configuration ---
# Check if a CUDA-enabled GPU is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of image-caption pairs to process together when generating embeddings
BATCH_SIZE = 32 # Adjust based on GPU memory for embedding generation

# ==============================================================================
# 2. Load Model and Processor
# ==============================================================================
# Print status message
print(f"Loading CLIP Model and Processor from: {MODEL_DIR}...")
# Start a try block to handle potential errors during loading
try:
    # Check if the specified model directory exists
    if not os.path.isdir(MODEL_DIR): raise FileNotFoundError("Model directory not found")
    # Load the fine-tuned CLIP model from the specified directory, move it to the GPU/CPU, and set it to evaluation mode (disables dropout, etc.)
    model = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    # Start another try block specifically for the processor
    try:
        # Attempt to load the processor saved alongside the fine-tuned model
        processor = CLIPProcessor.from_pretrained(MODEL_DIR)
        print("Loaded processor from fine-tuned model directory.")
    # If loading from the model directory fails...
    except:
        # Print a warning
        print("Could not load processor from model dir, loading from base...")
        # Load the original processor associated with the base CLIP model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Print success message if model and processor loaded
    print("✅ Model and Processor loaded.")
# If any exception occurs during loading...
except Exception as e:
    # Print a fatal error message
    print(f"❌ FATAL ERROR loading model/processor: {e}")
    # Stop the script execution
    exit()

# ==============================================================================
# 3. Create Dataset for Embedding Generation
# ==============================================================================
# Define a custom PyTorch Dataset class to handle loading data for embedding
class EmbeddingDataset(Dataset):
    # Constructor: Takes the path to the metadata file and the base image directory
    def __init__(self, metadata_path, image_base_dir):
        # Store the base image directory path
        self.image_base_dir = image_base_dir
        # Initialize an empty list to store the valid metadata items
        self.metadata = []
        # (valid_indices is defined but not strictly necessary with how metadata is appended now)
        self.valid_indices = []
        # Print status message
        print(f"Loading metadata from: {metadata_path}")
        # Start a try block for file reading
        try:
            # Open the metadata JSON Lines file for reading
            with open(metadata_path, 'r') as f:
                # Read all non-empty lines and parse each line as JSON into a list
                raw_metadata = [json.loads(line) for line in f if line.strip()]
        # If the metadata file doesn't exist...
        except FileNotFoundError:
            # Print error and re-raise the exception to stop the script
            print(f"❌ ERROR: Metadata file not found: {metadata_path}")
            raise
        # If the file contains invalid JSON...
        except json.JSONDecodeError as e:
             # Print error and re-raise the exception
             print(f"❌ ERROR: Invalid JSON in metadata file: {e}")
             raise

        # Print status message
        print("Verifying image paths...")
        # Iterate through the loaded raw metadata with a progress bar
        for i, item in enumerate(tqdm(raw_metadata, desc="Checking data")):
             # Check if the required keys 'image_path' and 'caption' are present in the item
            if "image_path" not in item or "caption" not in item:
                # Print warning and skip this item if keys are missing
                print(f"Warning: Skipping item {i}, missing 'image_path' or 'caption'.")
                continue
            # Construct the full path to the image file using the base directory and the filename part of the image_path
            img_path = os.path.join(self.image_base_dir, os.path.basename(item["image_path"]))
            # Check if the image file actually exists on disk
            if os.path.exists(img_path):
                # If it exists, add the metadata item to the final list for the dataset
                self.metadata.append(item)
                # (Appending to valid_indices is redundant here)
                self.valid_indices.append(i)
            # If the image file doesn't exist...
            else:
                 # Print a warning and skip this item
                 print(f"Warning: Image file not found for item {i}, skipping: {img_path}")

        # Print the final count of valid items loaded
        print(f"Loaded {len(self.metadata)} valid image-caption pairs.")
        # If no valid data was loaded...
        if not self.metadata:
             # Print error and stop the script
             print("❌ ERROR: No valid data loaded from metadata file.")
             exit()

    # Method required by PyTorch Dataset: Returns the total number of items in the dataset
    def __len__(self):
        return len(self.metadata)

    # Method required by PyTorch Dataset: Returns a single item (image path, caption, metadata) at the given index `idx`
    def __getitem__(self, idx):
        # Get the metadata dictionary for the requested index
        item = self.metadata[idx]
        # Construct the full image path again
        image_path = os.path.join(self.image_base_dir, os.path.basename(item["image_path"]))
        # Get the caption text
        caption = item["caption"]
        # Return a dictionary containing the image path, caption, and the original metadata item (useful for saving later)
        return {"image_path": image_path, "text": caption, "original_metadata": item}

# ==============================================================================
# 4. Collate Function for Embedding Generation
# ==============================================================================
# Get image preprocessing configuration details from the loaded CLIP processor
image_processor_config = processor.image_processor
# Get the mean values used for normalizing images
image_mean = image_processor_config.image_mean
# Get the standard deviation values used for normalizing images
image_std = image_processor_config.image_std
# Manually define image transformations based on processor config (resize, crop, tensor conversion, normalization)
# Determine the target image size (handling potential dictionary format for size)
image_size = image_processor_config.size['shortest_edge'] if isinstance(image_processor_config.size, dict) else image_processor_config.size
# Define the sequence of image transformations
image_transforms = transforms.Compose([
    transforms.Resize(image_size), # Resize the image
    transforms.CenterCrop(image_size), # Crop the center to the target size
    transforms.ToTensor(), # Convert the PIL Image to a PyTorch Tensor (pixels 0-1)
    transforms.Normalize(mean=image_mean, std=image_std), # Normalize pixel values
])

# Define the function that groups individual dataset items into batches for the DataLoader
def collate_fn_embed(batch):
    # Extract image paths from the batch items
    image_paths = [item['image_path'] for item in batch]
    # Extract text captions from the batch items
    texts = [item['text'] for item in batch]
    # Extract the original metadata dictionaries from the batch items
    original_metadata_batch = [item['original_metadata'] for item in batch]

    # Initialize list to hold processed image tensors
    processed_images = []
    # Initialize list to track indices of successfully loaded images within the batch
    valid_indices = []
    # Loop through the image paths in the current batch
    for i, img_path in enumerate(image_paths):
        # Start a try block for image loading and processing
        try:
            # Open the image file, convert to RGB
            img = Image.open(img_path).convert("RGB")
            # Apply the defined image transformations (resize, crop, tensor, normalize)
            processed_images.append(image_transforms(img))
            # If successful, record the index 'i' as valid
            valid_indices.append(i)
        # If any error occurs during image loading/processing...
        except Exception as e:
            # Print a warning and skip this image
            print(f"Warning: Skipping image during batch load: {img_path}. Error: {e}")

    # If NO images in the batch could be loaded...
    if not valid_indices:
        # Return None to signal the DataLoader to skip this entire batch
        return None

    # Filter the original texts list to keep only those corresponding to valid images
    filtered_texts = [texts[i] for i in valid_indices]
    # Filter the original metadata list similarly
    filtered_metadata = [original_metadata_batch[i] for i in valid_indices]

    # Process the filtered text captions using the CLIP processor (tokenize, pad, etc.)
    text_inputs = processor(text=filtered_texts, return_tensors="pt", padding=True, truncation=True)

    # Return a dictionary containing the batch data ready for the model
    return {
        "pixel_values": torch.stack(processed_images), # Stack the valid image tensors into a single batch tensor
        "input_ids": text_inputs['input_ids'], # Tokenized text IDs
        "attention_mask": text_inputs['attention_mask'], # Attention masks for text
        "metadata": filtered_metadata # Corresponding metadata for the valid items
    }

# ==============================================================================
# 5. Generate Embeddings
# ==============================================================================
# Print status message
print("Preparing DataLoader...")
# Create an instance of the EmbeddingDataset
dataset = EmbeddingDataset(METADATA_PATH, IMAGE_BASE_DIR)
# Create a PyTorch DataLoader to handle batching and loading data efficiently
# shuffle=False ensures embeddings are generated in a predictable order
# num_workers=0 avoids potential issues with multiprocessing and Google Drive access in Colab
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_embed, num_workers=0)

# Initialize empty lists to store the generated embeddings
all_image_embeddings = []
all_text_embeddings = []
# Initialize an empty list to store metadata in the order embeddings are generated (crucial for mapping)
ordered_metadata = []

# Print status message
print(f"Generating embeddings for {len(dataset)} items...")
# Disable gradient calculations (not needed for inference, saves memory)
with torch.no_grad():
    # Iterate through batches provided by the DataLoader with a progress bar
    for batch in tqdm(dataloader, desc="Generating Embeddings"):
        # If the collate function skipped the batch (e.g., all images failed to load), continue to the next
        if batch is None:
            continue

        # Move the image tensor batch to the GPU/CPU
        pixel_values = batch['pixel_values'].to(DEVICE)
        # Move the text ID tensor batch to the GPU/CPU
        input_ids = batch['input_ids'].to(DEVICE)
        # Move the text attention mask tensor batch to the GPU/CPU
        attention_mask = batch['attention_mask'].to(DEVICE)

        # Get image embeddings from the CLIP model
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        # Get text embeddings from the CLIP model
        text_outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # Normalize the image embeddings (important for similarity search)
        image_outputs /= image_outputs.norm(dim=-1, keepdim=True)
        # Normalize the text embeddings
        text_outputs /= text_outputs.norm(dim=-1, keepdim=True)

        # Move embeddings back to CPU, convert to NumPy arrays, and append to the master lists
        all_image_embeddings.append(image_outputs.cpu().numpy())
        all_text_embeddings.append(text_outputs.cpu().numpy())
        # Add the metadata dictionaries from this successfully processed batch to the ordered list
        ordered_metadata.extend(batch['metadata'])

# Print status message
print("Concatenating embeddings...")
# Check if any embeddings were actually generated
if not all_image_embeddings or not all_text_embeddings:
     # Print error and exit if lists are empty
     print("❌ ERROR: No embeddings were generated. Check for image loading errors.")
     exit()

# Combine the lists of batch embeddings into single large NumPy arrays, ensuring float32 type for FAISS
image_embeddings_np = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
text_embeddings_np = np.concatenate(all_text_embeddings, axis=0).astype(np.float32)

# Print the final shapes (number of items, embedding dimension)
print(f"Generated {image_embeddings_np.shape[0]} image embeddings with dimension {image_embeddings_np.shape[1]}.")
print(f"Generated {text_embeddings_np.shape[0]} text embeddings with dimension {text_embeddings_np.shape[1]}.")
# Print the number of metadata items collected
print(f"Metadata entries collected: {len(ordered_metadata)}")

# Sanity check: Ensure the number of embeddings matches the number of metadata entries collected
if image_embeddings_np.shape[0] != len(ordered_metadata) or text_embeddings_np.shape[0] != len(ordered_metadata):
    # Print a warning if there's a mismatch (indicates some items were skipped during processing)
    print("⚠️ Warning: Mismatch between number of embeddings and metadata entries! This indicates errors during processing.")
    # Consider adding exit() here if a perfect match is critical

# ==============================================================================
# 6. Save Embeddings and Ordered Metadata
# ==============================================================================
# Print status message and save the NumPy array of image embeddings to a .npy file
print(f"Saving Image Embeddings to: {IMAGE_EMBEDDINGS_FILE}...")
np.save(IMAGE_EMBEDDINGS_FILE, image_embeddings_np)
# Print status message and save the NumPy array of text embeddings to a .npy file
print(f"Saving Text Embeddings to: {TEXT_EMBEDDINGS_FILE}...")
np.save(TEXT_EMBEDDINGS_FILE, text_embeddings_np)
# Print status message and save the ordered metadata list to a JSON Lines file
print(f"Saving Metadata Order to: {METADATA_ORDER_FILE}...")
with open(METADATA_ORDER_FILE, 'w') as f:
     # Write each metadata item as a separate JSON line
     for item in ordered_metadata:
         f.write(json.dumps(item) + '\n')
# Print success message
print("✅ Embeddings and metadata order saved.")

# ==============================================================================
# 7. Build and Save FAISS Indexes
# ==============================================================================
# Get the dimension (length) of the embeddings from the NumPy array shape
dimension = image_embeddings_np.shape[1]

# --- Build Image Index ---
# Print status message
print("Building FAISS index for Images...")
# Create a simple, flat FAISS index using L2 distance (Euclidean distance)
image_index = faiss.IndexFlatL2(dimension)
# Print status message
print(f"Adding {image_embeddings_np.shape[0]} image vectors to index...")
# Add all the image embeddings to the FAISS index
image_index.add(image_embeddings_np)
# Print the total number of vectors now in the index
print(f"Total vectors in image index: {image_index.ntotal}")
# Print status message and save the FAISS index to the specified file
print(f"Saving Image Index to: {IMAGE_INDEX_FILE}...")
faiss.write_index(image_index, IMAGE_INDEX_FILE)
# Print success message
print("✅ Image Index built and saved.")

# --- Build Text Index ---
# Print status message
print("Building FAISS index for Texts...")
# Create another flat L2 index for text embeddings
text_index = faiss.IndexFlatL2(dimension)
# Print status message
print(f"Adding {text_embeddings_np.shape[0]} text vectors to index...")
# Add all the text embeddings to this index
text_index.add(text_embeddings_np)
# Print the total number of vectors in the text index
print(f"Total vectors in text index: {text_index.ntotal}")
# Print status message and save the text FAISS index to its file
print(f"Saving Text Index to: {TEXT_INDEX_FILE}...")
faiss.write_index(text_index, TEXT_INDEX_FILE)
# Print success message
print("✅ Text Index built and saved.")

# Print final summary message
print("\n--- Process Complete ---")
print(f"Embeddings (.npy) and FAISS indexes (.index) saved in: {OUTPUT_DIR}")
# Remind the user which metadata file to use for mapping in the backend API
print(f"Make sure to use '{os.path.basename(METADATA_ORDER_FILE)}' for mapping in your backend!")