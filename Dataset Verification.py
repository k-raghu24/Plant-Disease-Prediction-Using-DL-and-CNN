import os

dataset_path = r"C:\Users\krrag\OneDrive\Desktop\Projects\Plant disease predictiom\archive\PlantVillage"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        image_count = len(os.listdir(folder_path))
        print(f"{folder} --> {image_count} images")