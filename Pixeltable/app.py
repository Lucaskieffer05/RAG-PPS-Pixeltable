# This file now contains code that would typically be in a Jupyter notebook
# Import necessary modules
import os
import sys
from image_video_processor import ImageProcessor, VideoProcessor
import PIL.Image
import glob

# Define parameters (instead of command-line arguments)
# These would be set in separate cells in a Jupyter notebook
images_directory = "Media/Images/cars"
videos_directory = "Media/Videos"
search_query = "white car"  # Example query
search_image_path = "Media/Images/sample.jpg"  # Example path to a query image

print("Pixeltable Media Processing Demo")
print("===============================\n")

# Ensure directories exist
if not os.path.exists(images_directory):
    print(f"Warning: Images directory {images_directory} not found. Creating it.")
    os.makedirs(images_directory, exist_ok=True)
    
if not os.path.exists(videos_directory):
    print(f"Warning: Videos directory {videos_directory} not found. Creating it.")
    os.makedirs(videos_directory, exist_ok=True)

# Process images
print("\n== Testing Image Processor ==")
# List available images
image_files = glob.glob(os.path.join(images_directory, "*.*"))
print(f"Found {len(image_files)} images in {images_directory}")
for img in image_files[:5]:  # Show first 5 only
    print(f" - {os.path.basename(img)}")
if len(image_files) > 5:
    print(f" ... and {len(image_files) - 5} more")

# Initialize and set up the image processor
image_processor = ImageProcessor(directory=images_directory)
image_processor.setup_processing(extract_text=True, enable_search=True)

# Test search with text query
print(f"\nSearching for images matching text: '{search_query}'")
text_search_results = image_processor.search("Text", text_query=search_query)
print(f"Found {len(text_search_results)} matching images")

# Test search with image query
if os.path.exists(search_image_path):
    query_image = PIL.Image.open(search_image_path)
    print(f"\nSearching for images similar to: {os.path.basename(search_image_path)}")
    image_search_results = image_processor.search("Image", image_query=query_image)
    print(f"Found {len(image_search_results)} similar images")
else:
    print(f"\nQuery image not found at: {search_image_path}")

# Process videos
print("\n== Testing Video Processor ==")
# List available videos
video_files = glob.glob(os.path.join(videos_directory, "*.*"))
print(f"Found {len(video_files)} videos in {videos_directory}")
for vid in video_files[:5]:  # Show first 5 only
    print(f" - {os.path.basename(vid)}")
if len(video_files) > 5:
    print(f" ... and {len(video_files) - 5} more")

# Initialize and set up the video processor
video_processor = VideoProcessor(directory=videos_directory)
video_processor.setup_processing(extract_text=True, enable_search=True)

# Test search with text query
print(f"\nSearching for video frames matching text: '{search_query}'")
video_text_results = video_processor.search("Text", text_query=search_query)
print(f"Found {len(video_text_results)} matching frames")

# Test search with image query
if os.path.exists(search_image_path):
    query_image = PIL.Image.open(search_image_path)
    print(f"\nSearching for video frames similar to: {os.path.basename(search_image_path)}")
    video_image_results = video_processor.search("Image", image_query=query_image)
    print(f"Found {len(video_image_results)} similar frames")
else:
    print(f"\nQuery image not found at: {search_image_path}")

print("\nProcessing completed successfully")
