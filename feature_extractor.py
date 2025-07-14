import torch
import clip
from tqdm import tqdm
from PIL import Image
import numpy as np

def load_clip_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def extract_features(image_paths, model, preprocess, device, batch_size=512):
    from concurrent.futures import ProcessPoolExecutor
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
            # Fast, parallel loading and preprocessing on CPU
            with ProcessPoolExecutor() as pool:
                batch_images_preprocessed = list(pool.map(load_and_preprocess, batch_paths))
            valid = [img for img in batch_images_preprocessed if img is not None]
            valid_idx = [j for j, img in enumerate(batch_images_preprocessed) if img is not None]
            valid_paths.extend([batch_paths[j] for j in valid_idx])
            if not valid:
                continue
            batch_tensors = torch.stack(valid).to(device)
            features = model.encode_image(batch_tensors)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features), valid_paths
