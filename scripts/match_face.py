import os
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import torch

# Paths
EMBEDDING_FILE = 'embeddings.pkl'
INPUT_IMAGE_PATH = 'input.png'  # You can change this to whatever

# Load MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings
with open(EMBEDDING_FILE, 'rb') as f:
    celeb_embeddings = pickle.load(f)

# Load and process input image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise Exception("No face detected!")
    face = face.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        embedding = model(face).squeeze(0).numpy()
    return embedding

# Match input embedding to celeb embeddings
def find_top_matches(input_embedding, celeb_embeddings, top_k=3):
    similarities = {}
    for celeb, embeds in celeb_embeddings.items():
        celeb_avg = np.mean(embeds, axis=0)
        sim = cosine_similarity([input_embedding], [celeb_avg])[0][0]
        similarities[celeb] = sim

    # Sort by similarity (higher is better for cosine)
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches

# Main
if __name__ == '__main__':
    try:
        input_embedding = preprocess_image(INPUT_IMAGE_PATH)
        top_matches = find_top_matches(input_embedding, celeb_embeddings)
        print("\nTop Matches:")
        for name, score in top_matches:
            print(f"{name}: {score:.4f}")
    except Exception as e:
        print(f"[X] Failed: {e}")
