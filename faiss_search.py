import os
import numpy as np
import faiss
from tqdm import tqdm
from feature_extractor import extract_features

def get_all_image_paths(directory):
    import glob
    return glob.glob(f"{directory}/**/*.jpg", recursive=True)

def build_faiss_index(features, device="cpu"):
    dim = features.shape[1]
    index = faiss.IndexFlatIP(dim)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features.astype("float32"))
    return index

def get_or_build_index(dataset_path, model, preprocess, faiss_index_path, image_paths_path, batch_size=128, device="cpu"):
    if os.path.exists(faiss_index_path) and os.path.exists(image_paths_path):
        print("Loading FAISS index and image paths...")
        faiss_index = faiss.read_index(faiss_index_path)
        if device == "cuda":
            res = faiss.StandardGpuResources()
            try:
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            except:
                pass
        image_paths = np.load(image_paths_path, allow_pickle=True).tolist()
    else:
        print("Building FAISS index...")
        all_paths = get_all_image_paths(dataset_path)
        features, image_paths = extract_features(all_paths, model, preprocess, device, batch_size)
        faiss_index = build_faiss_index(features, device)
        if hasattr(faiss_index, 'index_cpu'):
            faiss.write_index(faiss_index.index_cpu(), faiss_index_path)
        else:
            faiss.write_index(faiss_index, faiss_index_path)
        np.save(image_paths_path, np.array(image_paths))
    return faiss_index, image_paths

def search_similar_images(query_image, model, preprocess, faiss_index, image_paths, device, top_k=6):
    with torch.no_grad():
        query_tensor = preprocess(query_image).unsqueeze(0).to(device)
        q_feat = model.encode_image(query_tensor)
        q_feat /= q_feat.norm(dim=-1, keepdim=True)
        q_feat = q_feat.cpu().numpy().astype("float32")
    D, I = faiss_index.search(q_feat, top_k)
    return [image_paths[i] for i in I[0]]
