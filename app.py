import streamlit as st
import os
import torch
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms


# ---- Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---- Load embeddings ----
with open('embeddings.pkl', 'rb') as f:
    celeb_data = pickle.load(f)  # { 'celeb_name': [emb1, emb2, ...] }

# ---- Streamlit UI ----
st.title("🙂 Celebrity Face Matcher")
st.markdown("Upload your image and see which celeb you look like!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ---- Functions ----

def extract_face(img):
    face = mtcnn(img)
    if face is not None:
        return face.unsqueeze(0).to(device)
    return None

def get_embedding(face_tensor):
    with torch.no_grad():
        return model(face_tensor).cpu().numpy()[0]  # shape: (512,)

def get_top_matches(target_embedding, celeb_data, top_k=3):
    similarities = {}

    for celeb, emb_list in celeb_data.items():
        celeb_avg = np.mean(emb_list, axis=0)  # shape: (512,)
        sim = cosine_similarity([target_embedding], [celeb_avg])[0][0]
        similarities[celeb] = sim

    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches

import torchvision.transforms as transforms

def tensor_to_pil(face_tensor):
    denormalize = transforms.Normalize(
        mean=[-1, -1, -1],
        std=[2, 2, 2]  # Because [-1,1] → [0,1]
    )
    img = denormalize(face_tensor.squeeze(0)).clamp(0, 1)  # remove batch dim and clamp to [0,1]
    return transforms.ToPILImage()(img.cpu())


# ---- Run on Upload ----
from PIL import Image, ImageOps

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = ImageOps.exif_transpose(img)  # Auto-rotate based on EXIF
        img.thumbnail((800, 800))           # Resize large images

        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Analyzing...")

        face_tensor = extract_face(img)

        if face_tensor is None:
            st.error("😥 Couldn't detect a face. Try another image.")
        else:
            embedding = get_embedding(face_tensor)
            top_matches = get_top_matches(embedding, celeb_data)

            # Show detected face image (optional)
            
            face_img = tensor_to_pil(face_tensor)
            st.image(face_img, caption='Detected Face', use_column_width=False)


            st.subheader("✨ Top Matches:")
            for name, score in top_matches:
                img_path = f"data/preprocessed_data/{name}/0.jpg"  # Adjust if needed
                celeb_img = Image.open(img_path).resize((160, 160))
                st.image(celeb_img, caption=f"{name} (Similarity: {score:.2f})", width=160)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
