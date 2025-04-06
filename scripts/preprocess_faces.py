import os
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Input and output directories
RAW_DIR = "data/RAW"
PROCESSED_DIR = "data/preprocessed_data"
IMAGE_SIZE = 160  # could also use 224

# Create detector
mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=20, keep_all=False, post_process=True)

def process_images_for_celeb(celeb_name):
    input_folder = os.path.join(RAW_DIR, celeb_name)
    output_folder = os.path.join(PROCESSED_DIR, celeb_name)
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)

        try:
            img = Image.open(input_path).convert('RGB')
            face = mtcnn(img, save_path=output_path)
            if face is None:
                print(f"[X] No face detected in: {img_file}")
            

        except Exception as e:
            print(f"[!] Failed processing {img_file}: {e}")

def main():
    celeb_list = os.listdir(RAW_DIR)
    for celeb in tqdm(celeb_list):
        process_images_for_celeb(celeb)

if __name__ == "__main__":
    main()
