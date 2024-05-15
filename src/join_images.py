import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from natsort import natsorted

title_order = [
    'pythia-31m',
    'pythia-70m',
    'pythia-160m',
    'pythia-410m',
    'pythia-1.4b',
    'pythia-2.8b',
    'pythia-6.9b',
    'pythia-12b'
]

def find_images(directory, filename):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                image_paths.append(os.path.join(root, file))
    return natsorted(image_paths)

def get_sorting_key(key):
    # Extract the numeric value and suffix from the key
    parts = key.split('-')
    if len(parts) >= 2:
        num_parts = parts[1].split('.')
        numeric = float(''.join(filter(str.isdigit, num_parts[0])))
        suffix = ''.join(filter(str.isalpha, num_parts[0]))
        if suffix == 'b':
            # Convert 'b' suffix to a large value for sorting
            suffix = 1000
        elif suffix == 'm':
            # Convert 'm' suffix to a smaller value for sorting
            suffix = 100
        else:
            suffix = 0
        return (numeric, suffix)
    else:
        return (0, 0)

def get_title(image_path):
    # Get the name 3 levels above the filename
    path_parts = os.path.normpath(image_path).split(os.sep)
    if len(path_parts) >= 4:
        title = path_parts[-4]
    else:
        title = os.path.basename(image_path)
    return title

def create_image_collage(directory, image_paths, output_filename):
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found with the specified filename.")
        return

    title_to_path = {get_title(image_path): image_path for image_path in image_paths}
    title_to_path = {title: title_to_path[title] for title in reversed(title_order) if title in title_to_path}

    num_cols = min(num_images, 4)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.tight_layout(pad=2.0)

    for i, (title, image_path) in enumerate(title_to_path.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.axis('off')
        
        ax.set_title(title)

    for i in range(i+1, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    save_pth = os.path.join(directory, output_filename)
    plt.savefig(save_pth, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Image collage saved as '{save_pth}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an image collage from matching filenames.")
    parser.add_argument("directory", help="Directory path to search for images.")
    parser.add_argument("filename", help="Filename to search for.")
    parser.add_argument("-o", "--output", default="image_collage.png", help="Output filename for the image collage.")

    args = parser.parse_args()

    # Find all images with the specified filename
    image_paths = find_images(args.directory, args.filename)

    # Create and save the image collage
    create_image_collage(args.directory, image_paths, args.output)



"""

python join_images.py \
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/figures/translation/dist/EleutherAI" \
    "Dataset_Content_(ds)_vs._Language_Model_(lm)_gens_-_log(en,_fr),_p=0.975.png"
    
"""