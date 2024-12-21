import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Load the ima
image = cv2.imread('data/set1_Image_01_40x_bf_02_fossicle_binary.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = "data/set1_Image_01_40x_bf_02_fossicle.png"  # Update the path to your image
image = Image.open(image_path)

# Define patch size
patch_width = 512  # Width of each patch
patch_height = 512  # Height of each patch

# Get the image dimensions
image_width, image_height = image.size

# Calculate the number of patches that fit horizontally and vertically
num_patches_x = image_width // patch_width
num_patches_y = image_height // patch_height

# Create a list to store patches
patches = []

# Generate patches systematically
for row in range(num_patches_y):
    for col in range(num_patches_x):
        # Define the top-left corner of the patch
        x_start = col * patch_width
        y_start = row * patch_height
        
        # Define the bottom-right corner of the patch
        x_end = x_start + patch_width
        y_end = y_start + patch_height
        
        # Crop the patch from the image
        patch = image.crop((x_start, y_start, x_end, y_end))
        
        # Append the patch and its coordinates to the list
        patches.append({
            "coordinates": (x_start, y_start, x_end, y_end),
            "image": patch
        })
output_dir = "patches_512x512"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
# Optional: Save patches as separate files
for idx, patch_info in enumerate(patches):
    patch_image = patch_info["image"]
    patch_image.save(f"{output_dir}/systematic_patch_{idx + 1}.png")  
# # Display the first 5 patches for verification
# for idx, patch_info in enumerate(patches[:5]):
#     patch_image = patch_info["image"]
#     coords = patch_info["coordinates"]
    
#     plt.figure(figsize=(5, 5))  # Adjust the figure size
#     plt.imshow(patch_image, cmap='gray')
#     plt.title(f"Patch {idx + 1} - Coordinates: {coords}")
#     plt.axis("on")
#     plt.show()