import os
import pickle
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch

# Paths
PREPROCESSED_DIR = 'data/preprocessed_data'
EMBEDDING_FILE = 'embeddings.pkl'

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image transformation: resize, to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
])

# Dictionary to store embeddings
embeddings_dict = {}

def generate_embeddings():
    for celeb in tqdm(os.listdir(PREPROCESSED_DIR), desc="Generating embeddings"):
        celeb_folder = os.path.join(PREPROCESSED_DIR, celeb)
        image_files = [f for f in os.listdir(celeb_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        celeb_embeddings = []

        for img_file in image_files:
            try:
                img_path = os.path.join(celeb_folder, img_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 160, 160]

                with torch.no_grad():
                    embedding = model(img_tensor)
                    celeb_embeddings.append(embedding.squeeze(0).numpy())  # convert to numpy

            except Exception as e:
                print(f"[X] Error processing {img_path}: {e}")

        if celeb_embeddings:
            embeddings_dict[celeb] = celeb_embeddings

    # Save as pickle
    with open(EMBEDDING_FILE, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"[✓] Saved embeddings to {EMBEDDING_FILE}")

if __name__ == "__main__":
    generate_embeddings()
