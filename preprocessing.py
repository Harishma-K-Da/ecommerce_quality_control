import os
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for root, dirs, files in os.walk(folder_path):  # walks through all subfolders
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    images.append(img)
                    filenames.append(filename)
    return images, filenames
