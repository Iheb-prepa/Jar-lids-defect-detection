import pandas as pd
import cv2
import json
import os

# Load CSV file
csv_path = 'cans_defect_detection_dataset_original/jarlids_annots.csv'  
df = pd.read_csv(csv_path)

# Create output directories if they don't exist
intact_dir = 'cans_defect_detection_dataset/intact'
damaged_dir = 'cans_defect_detection_dataset/damaged'

os.makedirs(intact_dir, exist_ok=True)
os.makedirs(damaged_dir, exist_ok=True)

# Iterate through the CSV rows
for index, row in df.iterrows():
    img_path = f"cans_defect_detection_dataset_original/{row['filename']}"  # Image path
    image = cv2.imread(img_path)  # Read the image
    
    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    # Extract bounding box data (region_shape_attributes column)
    region_shape = json.loads(row['region_shape_attributes'])
    x, y, width, height = region_shape['x'], region_shape['y'], region_shape['width'], region_shape['height']
    
    # Crop the image to the bounding box
    cropped_image = image[y:y+height, x:x+width]

    # Determine if it's 'intact' or 'damaged' from the region_attributes
    region_attributes = json.loads(row['region_attributes'])
    
    if region_attributes.get('type') == 'intact':
        output_dir = intact_dir
    else:
        output_dir = damaged_dir

    # Save the cropped image in the appropriate folder
    output_path = os.path.join(output_dir, f"{row['filename'].split('.')[0]}_bbox_{index}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Saved cropped image: {output_path}")
