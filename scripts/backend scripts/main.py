import fastapi
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles # For serving images
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import faiss
import torch
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
import numpy as np
import os
import json
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Configuration & Global Variables (Same as before)
# ==============================================================================
PROJECT_DIR = "/content/drive/My Drive/final year project"
MODEL_DIR = os.path.join(PROJECT_DIR, "clip_finetuned_culinary_cleaned_12345_then_6") # Your final best model path
FAISS_INDEX_DIR = os.path.join(PROJECT_DIR, "faiss_indexes_cleaned")
# *** IMPORTANT: Use the ORDERED metadata file from the indexing script ***
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata_order_cleaned.jsonl") # USE ORDERED METADATA
IMAGE_BASE_DIR = os.path.join(PROJECT_DIR, "Culinary_Caption_Dataset_Demo", "images")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
resources = {}

# ==============================================================================
# 2. Pydantic Models (*** UPDATED ImageSearchResponse ***)
# ==============================================================================
class TextQuery(BaseModel):
    query: str

class ImageSearchResult(BaseModel):
    image_path: str
    distance: float

# --- NEW RESPONSE MODEL FOR IMAGE SEARCH ---
class ImageSearchResponse(BaseModel):
    message: str
    results: list[ImageSearchResult]
# --- ---

class TextSearchResult(BaseModel):
    caption: str
    distance: float

class TextSearchResponse(BaseModel):
    message: str
    results: list[TextSearchResult]

# ==============================================================================
# 3. Load Resources on Startup (Dependency - Same as before)
# ==============================================================================
async def load_backend_resources():
    # ... (Loading logic remains exactly the same, ensures METADATA_PATH points to ordered file) ...
    # Load only if not already loaded
    if not resources:
        logger.info("--- Loading Backend Resources ---")
        try:
            # --- Load CLIP Model & Processor ---
            logger.info(f"Loading CLIP Model from: {MODEL_DIR}...")
            if not os.path.isdir(MODEL_DIR): raise FileNotFoundError("Model directory not found")
            resources['model'] = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()
            try:
                resources['processor'] = CLIPProcessor.from_pretrained(MODEL_DIR)
                logger.info("Loaded processor from fine-tuned model directory.")
            except:
                logger.warning("Could not load processor from model dir, loading from base...")
                resources['processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ Model and Processor loaded.")

            # --- Load FAISS Indexes ---
            logger.info(f"Loading FAISS Indexes from: {FAISS_INDEX_DIR}...")
            img_index_path = os.path.join(FAISS_INDEX_DIR, "images_cleaned.index")
            txt_index_path = os.path.join(FAISS_INDEX_DIR, "texts_cleaned.index")
            if not os.path.exists(img_index_path): raise FileNotFoundError("Image index not found")
            if not os.path.exists(txt_index_path): raise FileNotFoundError("Text index not found")
            resources['index_images'] = faiss.read_index(img_index_path)
            resources['index_texts'] = faiss.read_index(txt_index_path)
            logger.info(f"✅ Image Index loaded with {resources['index_images'].ntotal} vectors.")
            logger.info(f"✅ Text Index loaded with {resources['index_texts'].ntotal} vectors.")

            # --- Load Metadata Mapping (Ensure it's the ORDERED one) ---
            logger.info(f"Loading Metadata Mapping from: {METADATA_PATH}...")
            metadata_list = []
            with open(METADATA_PATH, 'r') as f: # Use the correct METADATA_PATH
                for line in f:
                    try: metadata_list.append(json.loads(line))
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line in metadata: {line.strip()}")
            resources['metadata_mapping'] = metadata_list
            logger.info(f"✅ Metadata mapping loaded for {len(metadata_list)} items.")
            if not metadata_list: raise ValueError("Metadata mapping is empty!")
            if len(metadata_list) != resources['index_images'].ntotal or len(metadata_list) != resources['index_texts'].ntotal:
                 logger.warning(f"Metadata count ({len(metadata_list)}) doesn't match index counts (Img: {resources['index_images'].ntotal}, Txt: {resources['index_texts'].ntotal}). ENSURE YOU ARE USING 'metadata_order_cleaned.jsonl'.")

            logger.info("--- Backend Resources Loaded Successfully ---")

        except Exception as e:
            logger.exception(f"❌ FATAL ERROR loading resources: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load backend resources: {e}")
    return resources


# ==============================================================================
# 4. FastAPI Application Instance & Static Files (Same as before)
# ==============================================================================
app = FastAPI(title="CLIP Culinary Search API")
static_parent_dir = os.path.dirname(IMAGE_BASE_DIR)
app.mount("/static_images", StaticFiles(directory=static_parent_dir), name="static_images")
logger.info(f"Serving static files from directory: {static_parent_dir} under /static_images")

# ==============================================================================
# 5. Helper Functions (Same as before)
# ==============================================================================
def get_text_embedding(text: str, model, processor):
    # ... (Remains the same) ...
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)

def get_image_embedding(image_pil: Image.Image, model, processor):
    # ... (Remains the same) ...
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype(np.float32)

# ==============================================================================
# 6. API Endpoints (*** UPDATED Return Values ***)
# ==============================================================================

@app.get("/")
async def root():
    return {"message": "CLIP Culinary Search Backend is running!", "status": "OK"}

# --- UPDATED response_model ---
@app.post("/search_images", response_model=ImageSearchResponse)
async def search_images_endpoint(query: TextQuery, res: dict = Depends(load_backend_resources)):
    """
    Accepts a text query and returns a message and list of matching image paths.
    """
    logger.info(f"Received text-to-image query: '{query.query}'")
    try:
        query_embedding = get_text_embedding(query.query, res['model'], res['processor'])
        distances, indices = res['index_images'].search(query_embedding, TOP_K)

        results_list = []
        if indices.size > 0:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(res['metadata_mapping']):
                    results_list.append(ImageSearchResult(
                        image_path=res['metadata_mapping'][idx]["image_path"],
                        distance=float(distances[0][i])
                    ))

        logger.info(f"Found {len(results_list)} image results.")
        # --- UPDATED Return ---
        return ImageSearchResponse(
            message=f"Found {len(results_list)} images closely related to your query:",
            results=results_list
        )
        # --- ---
    except Exception as e:
        logger.exception(f"Error during image search for query '{query.query}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during image search")

@app.post("/search_text", response_model=TextSearchResponse)
async def search_text_endpoint(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    res: dict = Depends(load_backend_resources)):
    """
    Accepts an image and returns a message and list of relevant captions.
    """
    img_pil = None
    source = "Unknown"
    try:
        # ... (Image loading logic remains the same) ...
        if image:
            source = f"file upload '{image.filename}'"
            logger.info(f"Received image via file upload: {image.filename}")
            contents = await image.read()
            img_pil = Image.open(BytesIO(contents)).convert("RGB")
        elif image_url:
            source = f"URL '{image_url}'"
            logger.info(f"Received image via URL: {image_url}")
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="No 'image' file or 'image_url' form data provided")

        query_embedding = get_image_embedding(img_pil, res['model'], res['processor'])
        distances, indices = res['index_texts'].search(query_embedding, TOP_K)

        results_list = []
        if indices.size > 0:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(res['metadata_mapping']):
                     results_list.append(TextSearchResult(
                        caption=res['metadata_mapping'][idx]["caption"],
                        distance=float(distances[0][i])
                    ))

        logger.info(f"Found {len(results_list)} text results for image from {source}.")
        # --- UPDATED Return Message ---
        return TextSearchResponse(
            message=f"Found {len(results_list)} relevant descriptions for the provided image:",
            results=results_list
        )
        # --- ---
    #...(Error handling remains the same) ...
    except requests.exceptions.RequestException as e:
         logger.error(f"Error fetching image from URL '{image_url}': {e}")
         raise HTTPException(status_code=400, detail=f"Could not fetch image from URL: {e}")
    except UnidentifiedImageError:
         logger.error(f"Invalid or corrupted image data provided from {source}.")
         raise HTTPException(status_code=400, detail="Invalid or corrupted image data")
    except Exception as e:
        logger.exception(f"Error during text search for image from {source}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during text search")

# ==============================================================================
# 7. Run Application (using Uvicorn - instructions remain the same)
# ==============================================================================
# Save as main.py and run:
# uvicorn main:app --host 0.0.0.0 --port 5000 --reload