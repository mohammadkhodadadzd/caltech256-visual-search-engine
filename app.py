import gradio as gr
from feature_extractor import load_clip_model
from faiss_search import get_or_build_index, search_similar_images

# --- Configuration ---
MODEL_NAME = "ViT-B/32"
DATASET_PATH = "./256_ObjectCategories"
FAISS_INDEX_PATH = "./faiss_index.bin"
IMAGE_PATHS_PATH = "./image_paths.npy"
BATCH_SIZE = 256

# --- Load CLIP Model ---
model, preprocess, device = load_clip_model(MODEL_NAME)

# --- Prepare or Load FAISS Index ---
faiss_index, image_paths = get_or_build_index(
    DATASET_PATH, model, preprocess, 
    FAISS_INDEX_PATH, IMAGE_PATHS_PATH, 
    batch_size=BATCH_SIZE, device=device
)

def gradio_search(query_image):
    results = search_similar_images(
        query_image, model, preprocess, 
        faiss_index, image_paths, device, top_k=6
    )
    return results

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Caltech256 Visual Similarity Search")
    gr.Markdown(
        "Upload any image and find the most visually similar images in the Caltech256 dataset using OpenAI CLIP & FAISS."
    )
    with gr.Row():
        with gr.Column(scale=1):
            query_image = gr.Image(type="pil", label="Upload your image here")
            search_button = gr.Button("🔎 Search", variant="primary")
        with gr.Column(scale=2):
            results_gallery = gr.Gallery(label="Search Results", columns=3, object_fit="contain", height="auto")
    search_button.click(fn=gradio_search, inputs=query_image, outputs=results_gallery)
    gr.Examples(
        examples=[
            "examples/query.jpg"
        ],
        inputs=query_image,
        outputs=results_gallery,
        fn=gradio_search,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()