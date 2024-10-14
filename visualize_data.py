import pandas as pd
import cv2
import json
import pathlib

# Load CSV file
csv_path = 'cans_defect_detection_dataset_original/jarlids_annots.csv'  # Update with your actual path
df = pd.read_csv(csv_path)

# Iterate through the CSV rows
for index, row in df.iterrows():
    img_path = f"cans_defect_detection_dataset_original/{row['filename']}"  # Image path
    image = cv2.imread(img_path)  # Read the image

    # Extract bounding box data (region_shape_attributes column)
    region_shape = json.loads(row['region_shape_attributes'])
    x, y, width, height = region_shape['x'], region_shape['y'], region_shape['width'], region_shape['height']
    
    # Draw the rectangle (bounding box)
    top_left = (x, y)
    bottom_right = (x + width, y + height)
    color = (0, 255, 0)  # Green color for the rectangle
    thickness = 2  # Thickness of the rectangle
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Show the image with bounding box
    cv2.imshow(f'Image {row["filename"]}', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
