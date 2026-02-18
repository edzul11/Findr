import streamlit as st
import os
import torch
import faiss
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# --- 1. Configuración y Carga de Recursos (Se carga solo una vez) ---
# Se utiliza para la carga inicial del modelo y el índice existentes.
@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/clip-ViT-B-16", device=get_device())
    
    faiss_data_folder = "C:/Users/Edzul/OneDrive/Desktop/clip/faiss_data"
    metadata_folder = "C:/Users/Edzul/OneDrive/Desktop/clip/metadata"
    faiss_index_path = os.path.join(faiss_data_folder, "faiss_index.bin")
    metadata_path = os.path.join(metadata_folder, "metadata.json")

    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return model, index, metadata
    else:
        st.error("No se encontraron los archivos del índice de FAISS o los metadatos.")
        return None, None, None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Función para Actualizar la Base de Datos ---
def update_database(model, index, metadata_list, image_folder):
    """Busca y procesa nuevas imágenes, actualizando el índice y los metadatos."""
    st.info("Buscando nuevas imágenes para procesar...")
    
    processed_files = {os.path.basename(m['path']) for m in metadata_list}
    all_files = os.listdir(image_folder)
    new_files = [f for f in all_files if f.lower().endswith((".jpg", ".png", ".jpeg")) and f not in processed_files]

    if not new_files:
        st.info("No se encontraron nuevas imágenes.")
        return index, metadata_list

    st.info(f"Procesando {len(new_files)} nuevas imágenes...")
    new_image_paths = [os.path.join(image_folder, f) for f in new_files]
    new_imgs_pil = [Image.open(p).convert("RGB") for p in new_image_paths]
    
    # Generar embeddings para las nuevas imágenes
    new_emb_imgs = model.encode(new_imgs_pil, convert_to_tensor=True, is_image=True, normalize_embeddings=True).cpu().numpy().astype(np.float32)

    # Añadir los nuevos embeddings al índice de FAISS
    index.add(new_emb_imgs)
    
    # Añadir los nuevos metadatos a la lista
    for p in new_image_paths:
        metadata_list.append({"path": p, "filename": os.path.basename(p)})

    # Guardar los índices y metadatos actualizados
    faiss_data_folder = "C:/Users/Edzul/OneDrive/Desktop/clip/faiss_data"
    metadata_folder = "C:/Users/Edzul/OneDrive/Desktop/clip/metadata"
    faiss_index_path = os.path.join(faiss_data_folder, "faiss_index.bin")
    metadata_path = os.path.join(metadata_folder, "metadata.json")

    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f)

    st.success("¡Base de datos actualizada con éxito!")
    return index, metadata_list

# --- 3. Interfaz de Usuario de Streamlit ---
st.title("Sistema de Búsqueda de Personas")
st.markdown("---")

image_folder = "C:/Users/Edzul/OneDrive/Desktop/clip/imagenes/personas_detectadas"
model, faiss_index, metadata_list = load_resources()

if model is not None and faiss_index is not None:
    # Botón para actualizar la base de datos
    if st.button("Actualizar Base de Datos"):
        faiss_index, metadata_list = update_database(model, faiss_index, metadata_list, image_folder)

    # Campo de texto para la consulta
    consulta = st.text_input("Describe a la persona que buscas:", "")
    
    buscar_btn = st.button("Buscar")

    if buscar_btn and consulta:
        emb_text = model.encode([consulta], convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().astype(np.float32)

        k = 3
        distances, indices = faiss_index.search(emb_text, k)

        st.subheader("Resultados de la Búsqueda:")

        cols = st.columns(3)
        for i, idx in enumerate(indices[0]):
            with cols[i]:
                if idx < len(metadata_list):
                    result_metadata = metadata_list[idx]
                    image_path = result_metadata['path']
                    st.image(image_path, caption=f"Score: {distances[0][i]:.4f}")
                    st.write(f"Nombre del archivo: {result_metadata['filename']}")
                else:
                    st.warning("No se encontraron suficientes resultados.")
else:
    st.warning("Por favor, asegúrate de que los archivos del índice de FAISS y los metadatos existan.")