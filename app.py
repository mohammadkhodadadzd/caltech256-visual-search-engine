#---- Step 1: Installation of Required Libraries ---
!pip install gradio -q
!pip install git+https://github.com/openai/CLIP.git -q
!pip install faiss-gpu-cu12  

# --- Step 2: Import Libraries ---
import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
from tqdm.notebook import tqdm
from time import time
import glob
import gradio as gr

print("Libraries installed and imported.")

#---- Step 2: Definition of the main variables and functions ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B/32"

DATASET_PATH = "/kaggle/input/caltech256/256_ObjectCategories"

FAISS_INDEX_PATH = "/kaggle/working/faiss_index.bin"
IMAGE_PATHS_PATH = "/kaggle/working/image_paths.npy"

print(f"Using device: {DEVICE}")
print(f"Dataset path: {DATASET_PATH}")
print(f"Model: {MODEL_NAME}")


model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

def get_all_image_paths(directory):
    
    return glob.glob(f"{directory}/**/*.jpg", recursive=True)

def extract_features(image_paths, batch_size=256):
    
    from concurrent.futures import ThreadPoolExecutor
    all_features = []
    valid_paths = []

    def load_and_preprocess(path):
        try:
            image = Image.open(path).convert("RGB")
            return preprocess(image)
        except Exception:
            return None

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            # parallel preprocessing
            with ThreadPoolExecutor() as pool:
                batch_images_preprocessed = list(pool.map(load_and_preprocess, batch_paths))
            # remove failed images
            valid = [img for img in batch_images_preprocessed if img is not None]
            valid_idx = [j for j, img in enumerate(batch_images_preprocessed) if img is not None]
            valid_paths.extend([batch_paths[j] for j in valid_idx])

            if not valid:
                continue
            batch_tensors = torch.stack(valid).to(DEVICE)
            features = model.encode_image(batch_tensors)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features), valid_paths

def build_faiss_index(features):
    "" "A FAISS Index for quick search based on cosine similarity." "" ""
    dimension = features.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(features.astype('float32'))
    return index

# --- Step 3: Search Index Preparation (run only once) ---

# Check if pre-computed index and paths exist

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(IMAGE_PATHS_PATH):
    print("Loading pre-computed FAISS index and image paths...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    image_paths = np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()
    print(f"Loaded {faiss_index.ntotal} vectors from index.")

else:
    print("Pre-computed files not found. Starting full indexing process...")
    # 1. Get all image paths
    all_paths = get_all_image_paths(DATASET_PATH)
    print(f"Found {len(all_paths)} total images.")
    
    # 2. Extract features and get valid paths
    start = time()
    features_np, image_paths = extract_features(all_paths)
    print(f"Extracted features for {len(image_paths)} valid images in {time() - start:.1f}s.")
    print(f"Feature vector shape: {features_np.shape}")

    # 3. Build FAISS index
    faiss_index = build_faiss_index(features_np)
    
    # 4. Save the index and paths for future runs
    print("Saving FAISS index and image paths to /kaggle/working/...")
    
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    np.save(IMAGE_PATHS_PATH, np.array(image_paths))
    print("Files saved successfully.")

# --- Step 2: Definition of the main function for the interface ---

def find_similar_images(query_image_pil, num_results=6):

    if query_image_pil is None:
        return [] 

    
    query_preprocessed = preprocess(query_image_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        query_features = model.encode_image(query_preprocessed)
        query_features /= query_features.norm(dim=-1, keepdim=True)

    query_np = query_features.cpu().numpy().astype('float32')
    distances, indices = faiss_index.search(query_np, num_results)
    
    result_paths = [image_paths[i] for i in indices[0]]
    return result_paths

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️Visual Search Engine")
    gr.Markdown("Upload an image to find similar images from the Caltech-256 dataset.")
    with gr.Row():
        with gr.Column(scale=1):
            query_image = gr.Image(type="pil", label="Upload your image here")
            search_button = gr.Button("🔎Search", variant="primary")
        with gr.Column(scale=2):
            results_gallery = gr.Gallery(label="Search results", columns=3, object_fit="contain", height="auto")
            
    search_button.click(fn=find_similar_images, inputs=query_image, outputs=results_gallery)
    
    gr.Examples(
        examples=[
            os.path.join(DATASET_PATH, "026.cake/026_0008.jpg"),
            os.path.join(DATASET_PATH, "008.bathtub/008_0006.jpg"),
            os.path.join(DATASET_PATH, "200.stained-glass/200_0010.jpg"),
        ],
        inputs=query_image,
        outputs=results_gallery,
        fn=find_similar_images,
        cache_examples=False 
    )

demo.launch(debug=True, allowed_paths=[DATASET_PATH])
