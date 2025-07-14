import argparse
import gradio as gr
from feature_extractor import load_clip_model
from faiss_search import get_or_build_index, search_similar_images

def main():
    parser = argparse.ArgumentParser(description="Visual Search Engine - CLIP + FAISS")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to your image dataset folder (e.g., ./256_ObjectCategories)")
    parser.add_argument("--faiss_index_path", type=str, required=True, help="Path to store/load the FAISS index (e.g., ./faiss_index.bin)")
    parser.add_argument("--image_paths_path", type=str, required=True, help="Path to store/load image paths file (e.g., ./image_paths.npy)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction (default: 256)")
    args = parser.parse_args()

    MODEL_NAME = "ViT-B/32"
    model, preprocess, device = load_clip_model(MODEL_NAME)
    faiss_index, image_paths = get_or_build_index(
        args.dataset_path, model, preprocess, 
        args.faiss_index_path, args.image_paths_path, 
        batch_size=args.batch_size, device=device
    )

    def gradio_search(query_image):
        return search_similar_images(
            query_image, model, preprocess, 
            faiss_index, image_paths, device, top_k=6
        )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Visual Search Engine")
        gr.Markdown(
            f"**Dataset:** `{args.dataset_path}`  \n"
            f"**FAISS index:** `{args.faiss_index_path}`  \n"
            f"**Image paths:** `{args.image_paths_path}`"
        )
        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(type="pil", label="Upload your image")
                search_button = gr.Button("🔎 Search", variant="primary")
            with gr.Column(scale=2):
                results_gallery = gr.Gallery(label="Results", columns=3, object_fit="contain", height="auto")
        search_button.click(fn=gradio_search, inputs=query_image, outputs=results_gallery)
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
    main()
