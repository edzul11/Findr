from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch, os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/clip-ViT-B-16", device=device)

# Carga imágenes
carpeta = "C:/Users/Edzul/OneDrive/Desktop/clip/imagenes"
imagenes = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
imgs_pil = [Image.open(p).convert("RGB") for p in imagenes]

# Genera embeddings
print(f"Procesando {len(imagenes)} imágenes...")
emb_imgs = model.encode(imgs_pil, convert_to_tensor=True, device=device, is_image=True, normalize_embeddings=True)

# Consulta
consulta = input("\nDescribe lo que buscas: ")
emb_text = model.encode([consulta], convert_to_tensor=True, device=device, normalize_embeddings=True)

# Similaridad
scores = util.cos_sim(emb_text, emb_imgs)[0]
best_idx = torch.argmax(scores).item()
print(f"\nMejor match: {imagenes[best_idx]} (score: {scores[best_idx]:.4f})")
print("CUDA disponible:", torch.cuda.is_available())