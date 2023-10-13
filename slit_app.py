import streamlit as st
import os
from PIL import Image
import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict
from transformers import CLIPModel, CLIPProcessor
import argparse
from tqdm.auto import tqdm
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
#import your_inference_module  # Import your image inference functions

st.title("Image Inference App")

# File Upload
uploaded_files = st.file_uploader("Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if os.path.exists("uploads"):
    os.rmdir("uploads")  # Remove the existing directory

# Create the directory
os.makedirs("uploads")

# Get the paths of all images in the given directory and return the image ids and their paths
def get_images_to_paths(image_directory, allowed_extensions):
    images_to_paths = {
        image_path.stem: image_path
        for image_path in image_directory.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    return images_to_paths, list(images_to_paths.keys())

# Generate CLIP embeddings for all images, handling damaged images if any
def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, regenerate_embeddings, embeddings_file):
    if not regenerate_embeddings:
        return set(), np.load(embeddings_file)

    damaged_image_ids, all_embeddings = set(), []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating CLIP embeddings")

    for i in range(0, len(all_image_ids), batch_size):
        batch_image_ids, batch_images = process_image_batch(all_image_ids, i, batch_size, images_to_paths, damaged_image_ids)
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        all_embeddings.extend(outputs.cpu().numpy())
        progress_bar.update(len(batch_image_ids))

    progress_bar.close()
    return damaged_image_ids, all_embeddings

# Process a batch of images, returning their ids and loaded images, while identifying damaged images
def process_image_batch(all_image_ids, start_idx, batch_size, images_to_paths, damaged_image_ids):
    batch_image_ids = all_image_ids[start_idx: start_idx + batch_size]
    batch_images = []

    for image_id in batch_image_ids:
        try:
            image = Image.open(images_to_paths[image_id])
            image.load()
            batch_images.append(image)
        except OSError:
            print(f"\nError processing image {images_to_paths[image_id]}, marking as corrupted.")
            damaged_image_ids.add(image_id)

    return batch_image_ids, batch_images

# Build an Annoy index using the generated CLIP embeddings
def build_annoy_index(all_embeddings):
    embeddings = np.array(all_embeddings)
    n_dimensions = embeddings.shape[1]

    annoy_index = AnnoyIndex(n_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(100)
    return annoy_index

# Compute the distance matrix of the embeddings using the Annoy index
def compute_distance_matrix(all_embeddings, annoy_index):
    n = len(all_embeddings)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            distance = annoy_index.get_distance(i, j)
            distances.append(distance)

    return distances

# Apply hierarchical clustering on the computed distance matrix with the given threshold
def apply_clustering(distances, threshold):
    condensed_distances = np.array(distances)
    Z = linkage(condensed_distances, method='average', optimal_ordering=True)
    return fcluster(Z, t=threshold, criterion='distance')

# Build clusters of image ids based on the clustering labels
def build_image_clusters(all_image_ids, labels):
    image_id_clusters = defaultdict(set)

    for image_id, cluster_label in zip(all_image_ids, labels):
        image_id_clusters[cluster_label].add(image_id)

    return image_id_clusters



clip_model = "openai/clip-vit-large-patch14-336"
threshold = 0.5
batch_size = 32

if uploaded_files:

    # Main function to process images and cluster them based on conceptual similarity using CLIP embeddings.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_file = 'embeddings.npy'
    regenerate_embeddings = True #check_and_load_embeddings(embeddings_file)

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model)
    allowed_extensions = {".jpeg", ".jpg", ".png", ".webp"}

    
    for uploaded_file in uploaded_files:
        # Save each uploaded image to the "uploads" folder
        image_path = os.path.join("uploads", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        #image = Image.open(image_path)
        #st.image(image, caption="Uploaded Image", use_column_width=True)
    image_directory = Path("uploads")
    images_to_paths, all_image_ids = get_images_to_paths(image_directory, allowed_extensions)
    damaged_image_ids, all_embeddings = generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, regenerate_embeddings, embeddings_file)

    if regenerate_embeddings:
        np.save(embeddings_file, all_embeddings)

    print("Building Annoy index...")
    annoy_index = build_annoy_index(all_embeddings)

    print("Computing distance matrix...")
    distances = compute_distance_matrix(all_embeddings, annoy_index)

    print("Applying hierarchical clustering...")
    labels = apply_clustering(distances, threshold)

    image_id_clusters = build_image_clusters(all_image_ids, labels)
    print(image_id_clusters)
    for idx, image_id_cluster in enumerate(image_id_clusters.values()):
        if len(image_id_cluster) < 2:
            continue
        for image_id in image_id_cluster:
            source = images_to_paths[image_id]
            label = f"cluster_{idx}"
            image = Image.open(source)
            st.markdown(f"**{label}**")

            st.image(image,  use_column_width=True)



    unique_image_ids = set(images_to_paths.keys()) - set(damaged_image_ids) - {image_id for cluster in image_id_clusters.values() for image_id in cluster if len(cluster) >= 2}
    for image_id in unique_image_ids:
        source = images_to_paths[image_id]
        label = "unique"
        image = Image.open(source)
        st.markdown(f"**{label}**")
        st.image(image,  use_column_width=True)
 

