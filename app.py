import gradio as gr
from feature_extractor import load_clip_model
from faiss_search import get_or_build_index, search_similar_images

# ---- User Input for Paths ----
import os

def start_app(dataset_path, faiss_index_path, image_paths_path):
    MODEL_NAME = "ViT-B/32"
    BATCH_SIZE = 256
    model, preprocess, device = load_clip_model(MODEL_NAME)
    faiss_index, image_paths = get_or_build_index(
        dataset_path, model, preprocess, 
        faiss_index_path, image_paths_path, 
        batch_size=BATCH_SIZE, device=device
    )
    def gradio_search(query_image):
        results = search_similar_images(
            query_image, model, preprocess, 
            faiss_index, image_paths, device, top_k=6
        )
        return results

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Visual Search Engine")
        gr.Markdown(
            f"**Dataset:** `{dataset_path}`  \n"
            f"**FAISS index:** `{faiss_index_path}`  \n"
            f"**Image paths:** `{image_paths_path}`"
        )
        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(type="pil", label="Upload your image")
                search_button = gr.Button("🔎 Search", variant="primary")
            with gr.Column(scale=2):
                results_gallery = gr.Gallery(label="Results", columns=3, object_fit="contain", height="auto")
        search_button.click(fn=gradio_search, inputs=query_image, outputs=results_gallery)

        # Example from assets
        gr.Examples(
            examples=[
                "assets/sample.png",
            ],
            inputs=query_image,
            outputs=results_gallery,
            fn=gradio_search,
            cache_examples=False
        )
    demo.launch()

if __name__ == "__main__":
    print("Welcome to Visual Search Engine!")
    dataset_path = input("Enter the dataset folder path (e.g., ./256_ObjectCategories): ").strip()
    while not os.path.isdir(dataset_path):
        dataset_path = input("Invalid folder. Enter the dataset folder path: ").strip()

    faiss_index_path = input("Enter the path to save/load FAISS index (e.g., ./faiss_index.bin): ").strip()
    image_paths_path = input("Enter the path to save/load image paths (e.g., ./image_paths.npy): ").strip()
    start_app(dataset_path, faiss_index_path, image_paths_path)
