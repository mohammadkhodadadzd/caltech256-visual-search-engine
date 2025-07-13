import torch
import clip
from tqdm import tqdm
from PIL import Image
import numpy as np

def load_clip_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def extract_features(image_paths, model, preprocess, device, batch_size=128):
    all_features = []
    valid_paths = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch = image_paths[i:i+batch_size]
            images = []
            batch_valid_paths = []
            for path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(preprocess(img))
                    batch_valid_paths.append(path)
                except:
                    continue
            if not images:
                continue
            img_tensor = torch.stack(images).to(device)
            feats = model.encode_image(img_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
    return np.vstack(all_features), valid_paths