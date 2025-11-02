import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import os # Import the os module to check for file existence

# ==============================================================================
# 1. Setup: Load Your Fine-Tuned Model
# ==============================================================================
print("Step 1: Loading the fine-tuned model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/content/drive/My Drive/final year project/clip_finetuned_food101_v1" 

# Check if the model directory exists
if not os.path.isdir(model_path):
    print(f"Error: Model directory not found at '{model_path}'")
    print("Please make sure you have trained the model and it's saved in the correct location.")
    exit()

# Load the processor and the fine-tuned model
processor = CLIPProcessor.from_pretrained(model_path)
model = CLIPModel.from_pretrained(model_path).to(device)
model.eval() # Set the model to evaluation mode permanently

# ==============================================================================
# 2. Prepare Candidate Descriptions (Done once)
# ==============================================================================
print("Step 2: Preparing candidate descriptions from Food-101 classes...")
food_classes = load_dataset("food101", split="train").features["label"].names

# Format the class names into descriptive sentences
candidate_descriptions = [f"a photo of {food.replace('_', ' ')}" for food in food_classes]

# ==============================================================================
# 3. Interactive Loop for Testing
# ==============================================================================
while True:
    # --- Prompt the user for an image path ---
    image_path = input("\nEnter the path to your image (or type 'quit' to exit): ")

    if image_path.lower() == 'quit':
        break

    # --- Check if the file exists and load the image ---
    if not os.path.exists(image_path):
        print(f"❌ Error: The file '{image_path}' was not found. Please try again.")
        continue # Skip to the next loop iteration

    try:
        input_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error: Could not open or process the image. Reason: {e}")
        continue

    # --- Perform Inference ---
    print("\nProcessing image... Please wait.")
    with torch.no_grad():
        inputs = processor(
            text=candidate_descriptions, 
            images=input_image, 
            return_tensors="pt", 
            padding=True
        ).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_match_index = probs.argmax().item()

    # --- Display the Result ---
    best_description = candidate_descriptions[best_match_index]
    best_probability = probs[0, best_match_index].item()

    print("\n--- Model Prediction ---")
    print(f"✅ Best Description: '{best_description}'")
    print(f"Confidence: {best_probability * 100:.2f}%")

print("\nExiting the program. Goodbye!")