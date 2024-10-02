import os
import numpy as np
from PIL import Image
import glob

def zscore_normalization(image_array):
    """
    Apply Z-score normalization to the image array.
    """
    mean = image.mean()
    std = image.std()
    image -= mean
    image /= max(std, 1e-8)
    return image

def load_and_save_bmp_as_npy(input_folder, output_folder, image_size=(256, 256)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of all BMP files in the input folder
    bmp_files = glob.glob(os.path.join(input_folder, '*.bmp'))
    
    for bmp_file in bmp_files:
        # Load the image
        image = Image.open(bmp_file)
        
        # Resize the image using LANCZOS for high-quality downsampling
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        # Apply Z-score normalization
        normalized_image = zscore_normalization(image_array)
        
        # Create the output file path
        base_name = os.path.basename(bmp_file)
        output_file = os.path.join(output_folder, os.path.splitext(base_name)[0] + '.npy')
        
        # Save the array as a .npy file
        np.save(output_file, normalized_image)
        
        print(f"Saved {output_file}")


def print_image_shapes(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out BMP files
    bmp_files = [file for file in files if file.endswith('.bmp')]
    
    # Loop over BMP files and print their shapes
    for bmp_file in bmp_files:
        # Construct the full file path
        file_path = os.path.join(directory, bmp_file)
        
        # Open the image
        with Image.open(file_path) as img:
            # Print the image's shape (width, height)
            print(f"{bmp_file}: {img.size} (Width x Height), Mode: {img.mode}")



def save_images_as_npy(directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out BMP files
    bmp_files = [file for file in files if file.endswith('.bmp')]
    
    # Loop over BMP files, print their shapes, and save as .npy
    for bmp_file in bmp_files:
        # Construct the full file path
        file_path = os.path.join(directory, bmp_file)
        
        # Open the image
        with Image.open(file_path) as img:
            # Convert image to a NumPy array
            img_array = np.array(img)
            
            # Print the image's shape
            print(f"{bmp_file}: {img_array.shape} (Height, Width, Channels)")
            
            # Create the output file path for .npy
            npy_file = os.path.join(output_directory, os.path.splitext(bmp_file)[0] + '.npy')
            
            # Save the NumPy array as .npy
            np.save(npy_file, img_array)
            
            print(f"Saved {bmp_file} as {npy_file}")


# Example usage
input_folder = '/path/to/data/Training Set/images/'
output_folder = '/path/to/data/Training Set/npy/'

#load_and_save_bmp_as_npy(input_folder, output_folder)
#print_image_shapes(input_folder)
save_images_as_npy(input_folder, output_folder)