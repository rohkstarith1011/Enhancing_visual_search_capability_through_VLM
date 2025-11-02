import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import json

# ==============================================================================
#  A. Configuration (*** UPDATED FOR NEXT CHUNK ***)
# ==============================================================================
# --- SELECT WHICH CLEANED CHUNK TO TRAIN ON NEXT ---
TRAIN_CHUNK_NUMBER = 7 # Train on CLEANED chunk 6
# --- ---

project_dir = "/content/drive/My Drive/final year project"
base_dataset_dir = os.path.join(project_dir, "Culinary_Caption_Dataset_Demo")
# --- Point to the CLEANED chunks directory ---
chunk_metadata_dir = os.path.join(base_dataset_dir, "metadata_cleaned_chunks") # USE CLEANED CHUNKS
# --- ---
image_base_dir = os.path.join(base_dataset_dir, "images")
# --- New Output Directory for this run ---
# Name reflects adding cleaned chunk 4 after the combined 1-3 run
output_dir = os.path.join(project_dir, f"clip_finetuned_culinary_cleaned_123456_then_{TRAIN_CHUNK_NUMBER}")
os.makedirs(output_dir, exist_ok=True)

# --- Load Model from the PREVIOUS Best Run (Combined 1-3 on cleaned data) ---
best_previous_model_dir = os.path.join(project_dir, "clip_finetuned_culinary_cleaned_12345_then_6") # Load from the last run
if os.path.isdir(best_previous_model_dir):
    model_load_path = best_previous_model_dir
    print(f"Continuing training on chunk {TRAIN_CHUNK_NUMBER}. Loading model from: {model_load_path}")
else:
    # Fallback only if the expected previous model isn't found
    model_load_path = "openai/clip-vit-base-patch32"
    print(f"Warning: Best previous model not found at {best_previous_model_dir}. Starting from original CLIP weights (may not be intended).")
# --- ---

# --- Adjust Training Params (Lower LR further) ---
num_epochs = 15 # Can keep epochs, early stopping will handle it
batch_size = 32
learning_rate = 1e-7 # Lower learning rate again for refinement (0.0000005)
weight_decay = 0.05
early_stopping_patience = 4
best_val_loss = float('inf')
patience_counter = 0
# --- ---

# ==============================================================================
#  B. Custom Dataset Class (Unchanged)
# ==============================================================================
class CustomImageCaptionDataset(Dataset):
    # ... (Dataset class remains exactly the same) ...
    def __init__(self, metadata_list, image_base_dir, transform=None):
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.metadata = metadata_list
        # print(f"Dataset initialized with {len(self.metadata)} items.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.image_base_dir, os.path.basename(item["image_path"]))
        try: image = Image.open(image_path).convert("RGB")
        except FileNotFoundError: return None
        except Exception as e: return None
        caption = item["caption"]
        if self.transform:
            try: image = self.transform(image)
            except Exception as e: print(f"E: transform img {image_path}: {e}"); return None
        return {"image": image, "text": caption}

# ==============================================================================
#  C. Load Data and Set Up Model (*** UPDATED FOR SINGLE CLEANED CHUNK ***)
# ==============================================================================
print("Setting up model and dataloaders...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained(model_load_path).to(device)
try:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Loaded processor from base model ID 'openai/clip-vit-base-patch32'.")
except Exception as e:
    print(f"W: Could not load processor from base ID. Trying from '{model_load_path}'. E: {e}")
    processor = CLIPProcessor.from_pretrained(model_load_path if os.path.isdir(model_load_path) else "openai/clip-vit-base-patch32")

image_size = processor.image_processor.size['shortest_edge']
# Transforms remain the same
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])
val_transforms = transforms.Compose([
    transforms.Resize(image_size), transforms.CenterCrop(image_size),
])

# --- Load Metadata ONLY for the selected CLEANED chunk ---
print(f"Loading CLEANED metadata from chunk: {TRAIN_CHUNK_NUMBER}...")
chunk_filename = f"metadata_cleaned_chunk_{TRAIN_CHUNK_NUMBER}.jsonl" # Use cleaned chunk filename
jsonl_path = os.path.join(chunk_metadata_dir, chunk_filename) # Point to cleaned chunks dir

all_metadata = []
try:
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            try: all_metadata.append(json.loads(line))
            except json.JSONDecodeError: print(f"W: Skip invalid JSON line {line_num+1} in {jsonl_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Cleaned metadata file not found: {jsonl_path}. Exiting.")
    exit()
except Exception as e:
    print(f"❌ ERROR: Failed to load metadata for cleaned chunk {TRAIN_CHUNK_NUMBER}. Error: {e}")
    exit()

if not all_metadata:
    print(f"❌ ERROR: Cleaned metadata for chunk {TRAIN_CHUNK_NUMBER} is empty. Exiting.")
    exit()
print(f"Loaded {len(all_metadata)} items from cleaned chunk {TRAIN_CHUNK_NUMBER}.")
# --- ---

# --- Splitting logic (70/30 on THIS chunk's metadata list) ---
# ... (Splitting logic remains the same, uses 'all_metadata' from the single chunk) ...
train_split_ratio = 0.7
total_size = len(all_metadata)
train_size = int(train_split_ratio * total_size)
val_size = max(1, total_size - train_size)
train_size = total_size - val_size

if train_size <= 0 or val_size <= 0:
    print(f"❌ ERROR: Not enough data ({total_size} items) in cleaned chunk {TRAIN_CHUNK_NUMBER} for split.")
    exit()

print(f"Splitting cleaned chunk {TRAIN_CHUNK_NUMBER} -> Train: {train_size}, Validation: {val_size}")
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]; val_indices = indices[train_size:]
train_metadata = [all_metadata[i] for i in train_indices]
val_metadata = [all_metadata[i] for i in val_indices]
# --- ---

# --- Create SEPARATE Dataset instances ---
train_dataset = CustomImageCaptionDataset(train_metadata, image_base_dir, transform=train_transforms)
validation_dataset = CustomImageCaptionDataset(val_metadata, image_base_dir, transform=val_transforms)
# --- ---

# --- Collate function and DataLoaders (Unchanged) ---
# ... (Remains the same as previous script) ...
normalize = transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    images = [item['image'] for item in batch]; texts = [item['text'] for item in batch]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    try: image_tensors = [normalize(transforms.ToTensor()(image)) for image in images]
    except Exception as e: print(f"E: collate_fn img proc: {e}"); return None
    if not image_tensors: return None
    return {"input_ids": text_inputs['input_ids'], "attention_mask": text_inputs['attention_mask'], "pixel_values": torch.stack(image_tensors)}

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True if device=='cuda' else False)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True if device=='cuda' else False)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ==============================================================================
#  D. Training and Validation Loop (Unchanged)
# ==============================================================================
print(f"Starting training on CLEANED chunk {TRAIN_CHUNK_NUMBER}...")
# ... (The entire training loop remains the same) ...
# ... (It will save to the new 'output_dir' defined in Section A) ...
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0; train_batches_processed = 0
    # ... (Inner training loop) ...
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
        if batch is None: continue
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True); loss = outputs.loss
            if torch.isnan(loss): print(f"W: NaN loss train {train_batches_processed+1}. Skip."); optimizer.zero_grad(); continue
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            total_train_loss += loss.item(); train_batches_processed += 1
        except Exception as e: print(f"\nE: train batch {train_batches_processed+1}: {e}"); optimizer.zero_grad()
    avg_train_loss = total_train_loss / train_batches_processed if train_batches_processed > 0 else float('nan')

    model.eval()
    total_val_loss = 0; total_correct_preds = 0; total_val_samples = 0; val_batches_processed = 0
    with torch.no_grad():
        # ... (Inner validation loop) ...
         for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            if batch is None: continue
            try:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs, return_loss=True); loss = outputs.loss
                if torch.isnan(loss): print(f"W: NaN loss val {val_batches_processed+1}. Skip."); continue
                total_val_loss += loss.item(); val_batches_processed += 1
                image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
                text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                image_features /= image_features.norm(dim=-1, keepdim=True); text_features /= text_features.norm(dim=-1, keepdim=True)
                if torch.isnan(image_features).any() or torch.isnan(text_features).any(): print(f"W: NaN feats val {val_batches_processed}. Skip acc."); continue
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predictions = similarity.argmax(dim=1); targets = torch.arange(len(predictions), device=device)
                correct_preds = (predictions == targets).sum()
                total_correct_preds += correct_preds.item(); total_val_samples += len(predictions)
            except Exception as e: print(f"\nE: val batch {val_batches_processed+1}: {e}")
    avg_val_loss = total_val_loss / val_batches_processed if val_batches_processed > 0 else float('nan')
    accuracy = (total_correct_preds / total_val_samples) * 100 if total_val_samples > 0 else 0

    print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    if not torch.isnan(torch.tensor(avg_val_loss)) and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss; patience_counter = 0
        print("Validation loss improved. Saving best model...")
        try: model.save_pretrained(output_dir); processor.save_pretrained(output_dir) # Save to the new output_dir
        except Exception as e: print(f"Error saving model/processor: {e}")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}) or was NaN. Patience: {patience_counter}/{early_stopping_patience}")

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered. Training finished.")
        break

print(f"Training complete for cleaned chunk {TRAIN_CHUNK_NUMBER}. Best model saved at: {output_dir}") # Updated print message