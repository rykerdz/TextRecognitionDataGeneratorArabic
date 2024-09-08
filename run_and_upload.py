import random
from datasets import load_dataset
from PIL import Image
from trdg.generators import GeneratorFromStrings
import boto3
from io import BytesIO
from multiprocessing import Process, Queue, Pool
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

# Function to generate images (batch processing)
def generate_batch(batch_args):
    strings_batch, fonts, queue, batch_size = batch_args
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
batch_queue = Queue(maxsize=50)

# Start the upload process
uploader = Process(target=upload_batch, args=(batch_queue,))
uploader.start()

# Function to parallelize image generation
def parallel_generate(strings, fonts, batch_size=1000, thread_count=8):
    queue = Queue(maxsize=50)
    pool = Pool(thread_count)

    # Prepare batch arguments
    batch_args = [
        (strings[i:i + batch_size], fonts, queue, batch_size)
        for i in range(0, len(strings), batch_size)
    ]

    # Map the batch arguments to the worker function
    pool.map(generate_batch, batch_args)
    
    pool.close()
    pool.join()
    
    return queue

# Load the dataset
ds = load_dataset("premio-ai/TheArabicPile_Articles", "original", split='train')
keywords_to_remove = {"العنوان:", "المقال:"}

offset = 0
# Process the dataset in top-level batches
for batch_num in range(0, len(ds)//1_000_000):
    print(f"Top-level Batch: {batch_num}")

    print("Processing strings")
    strings_list = []
    for i, item in enumerate(ds):
        if len(strings_list) >= 1_000_000:
            offset = i
            break  # Stop once we've reached 1 million sub-strings
        print(item)
        line = item['text']  # Ensure max 50 words
        sub_strings = process_line(line)
        strings_list.extend(sub_strings)
        
    random.shuffle(strings_list)
    strings_list = strings_list[:1_000_000]  # Ensure batch size is exactly 1 million

    print("Started Generating images")
    # Generate and upload images for this top-level batch
    batch_queue = parallel_generate(strings_list, fonts, batch_size=1000, thread_count=args.thread_count)

# Signal the uploader process to stop
batch_queue.put(None)
uploader.join()

print("All images processed and uploaded.")
