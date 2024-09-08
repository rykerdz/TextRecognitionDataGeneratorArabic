import random
from datasets import load_dataset
from PIL import Image
from trdg.generators import GeneratorFromStrings
import boto3
from io import BytesIO
from multiprocessing import Process, Queue
import time
import os

# Initialize the S3 client
s3_client = boto3.client('s3')

# Define your S3 bucket name
bucket_name = 'ocr88'

print("Loading fonts....")
font_dir = "/mnt/volume_nyc1_01/fonts/"
fonts = [
  os.path.join(font_dir, p)
  for p in os.listdir(font_dir)
  if os.path.splitext(p)[1] == ".ttf" or os.path.splitext(p)[1] == ".TTF"
]

print(f"Loaded {len(fonts)} Font!")

# Function to process a line of text
def process_line(line, keywords_to_remove):
    words = line.split()
    filtered_words = [word for word in words if word not in keywords_to_remove]

    sub_strings = []
    n = len(filtered_words)

    # Generate sub-strings of 3 to 6 words
    for i in range(n - 2):
        for length in range(3, 10):
            if i + length <= n:
                sub_string = " ".join(filtered_words[i:i+length])
                sub_strings.append(sub_string)

    return sub_strings

# Function to generate and upload images
def generate_and_upload(strings_batch, queue):
    generator = GeneratorFromStrings(
        strings_batch,
        size=100,
        fonts=fonts,
        count=len(strings_batch),
        language='ar',
        skewing_angle=15,
        random_skew=True,
        background_type=3,  # 3 image
        distorsion_type=1,
        distorsion_orientation=2,
        text_color='#000000,#FFFFFF',
        blur=0,
        random_blur=True,
        rtl=True,
        image_dir='/mnt/volume_nyc1_01/images/'
    )

    batch_size = 1000
    batch = []
    start_generate_time = time.time()
    
    for img, lbl in generator:
        batch.append((img, lbl))
        
        if len(batch) == batch_size:
            queue.put(batch)
            generate_time = time.time() - start_generate_time
            print(f"Generation and Queueing Time for one batch: {generate_time:.2f} seconds")
            batch = []
            start_generate_time = time.time()

    # Process any remaining images in the last batch
    if batch:
        queue.put(batch)
        generate_time = time.time() - start_generate_time
        print(f"Generation and Queueing Time for one batch: {generate_time:.2f} seconds")

# Function to process and upload a batch of images
def upload_batch(queue):
    while True:
        batch = queue.get()
        if batch is None:  # Sentinel to stop the process
            break

        start_upload_time = time.time()
        for img, lbl in batch:
            if img is not None:
                # Convert PIL image to bytes
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Define the S3 object name
                save_as = lbl.replace(" ", "_")
                object_name = f'generated_images/img_{img_byte_arr[200]}_{save_as}.png'

                # Upload the image to S3
                s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=img_byte_arr, ContentType='image/png')

        upload_time = time.time() - start_upload_time
        print(f"Upload Time for one batch: {upload_time:.2f} seconds")

# Create a Queue for batch processing
batch_queue = Queue(maxsize=10)

# Start the upload process
uploader = Process(target=upload_batch, args=(batch_queue,))
uploader.start()

# Load the dataset
ds = load_dataset("premio-ai/TheArabicPile_Articles", "original", split='train')
keywords_to_remove = {"العنوان:", "المقال:"}

# Process the dataset in top-level batches
for batch_num in range(0, len(ds), 1_000_000):
    strings_list = []

    for i in range(batch_num, min(batch_num + 1_000_000, len(ds))):
        line = ds[i]['text']  
        sub_strings = process_line(line, keywords_to_remove)
        strings_list.extend(sub_strings)
        
    random.shuffle(strings_list)
    strings_list = strings_list[:1_000_000]  # Ensure batch size is exactly 1 million
    
    # Generate and upload images for this top-level batch
    generate_and_upload(strings_list, batch_queue)

# Signal the uploader process to stop
batch_queue.put(None)
uploader.join()

print("All images processed and uploaded.")
