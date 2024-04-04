import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args

import Models

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


# Path to the folder containing the images
image_folder = str(input("/path/to/your/image/folder: "))


# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Counter for the number of images
image_count = 0

@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    def jit_predict(self, x):
        # Not actually JITed since this is a single shot script,
        # but this is the function you would decorate with @jax.jit
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    # Resize
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.BICUBIC,
        )
    return image


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def load_model_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> PredModel:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.msgpack",
        revision=revision,
        token=token,
    )

    model_config = hf_hub_download(
        repo_id=repo_id,
        filename="sw_jax_cv_config.json",
        revision=revision,
        token=token,
    )

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]


def get_tags(
    probs: Any,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(
            gen_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(
            char_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)


def main(opts: ScriptOptions):
    global image_path
    repo_id = MODEL_REPO_MAP.get(opts.model)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model, target_size = load_model_hf(repo_id=repo_id)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Loading image and preprocessing...")
    # get image
    img_input: Image.Image = Image.open(image_path)
    # ensure image is RGB
    img_input = pil_ensure_rgb(img_input)
    # pad to square with white background
    img_input = pil_pad_square(img_input)
    img_input = pil_resize(img_input, target_size)
    # convert to numpy array and add batch dimension
    inputs = np.array(img_input)
    inputs = np.expand_dims(inputs, axis=0)
    # NHWC image RGB to BGR
    inputs = inputs[..., ::-1]

    print("Running inference...")
    outputs = model.predict(inputs)

    print("Processing results...")
    caption, taglist, ratings, character, general = get_tags(
        probs=outputs,
        labels=labels,
        gen_threshold=opts.gen_threshold,
        char_threshold=opts.char_threshold,
    )

    print("--------")
    print(f"Caption: {caption}")
    print("--------")
    print(f"Tags: {taglist}")

    print("--------")
    print("Ratings:")
    for k, v in ratings.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"Character tags (threshold={opts.char_threshold}):")
    for k, v in character.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"General tags (threshold={opts.gen_threshold}):")
    for k, v in general.items():
        print(f"  {k}: {v:.3f}")

    print("Done!")
    return caption


if __name__ == "__main__":
    # Iterate through the image files
    for imageName in image_files:
        # Full path of the image file
        opts, _ = parse_known_args(ScriptOptions)
        if opts.model not in MODEL_REPO_MAP:
            print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
            raise ValueError(f"Unknown model name '{opts.model}'")
        
        image_path_variable = os.path.join(image_folder, imageName)
        image_path = Path(image_path_variable)
        Anime_Tags = main(opts)
        # Corresponding text file name for the image
        text_file_name = os.path.splitext(imageName)[0] + ".txt"
        
        with open(os.path.join(image_folder, text_file_name), "w") as text_file:
            # Write the description, including the full path of the image
            text_file.write(f"{Anime_Tags}")
        
        # Increment the image count
        image_count += 1
    print("Total number of images processed:", image_count)

show_top_tags = 50 
import collections
from collections import Counter
top_tags = Counter()

for txt in [f for f in os.listdir(image_folder) if f.lower().endswith(".txt")]:
  with open(os.path.join(image_folder, txt), 'r') as f:
    top_tags.update([s.strip() for s in f.read().split(",")])

top_tags = Counter(top_tags)
print("\\\\\\\\\\\\\\\\\\\\\\")
print(f"📊 Top {show_top_tags} tags:")
for k, v in top_tags.most_common(show_top_tags):
  print(f"{k} ({v})")
print("thanks to Hollowstrawberry, SmilingWolf, and neggles whos code helped in making this script")

deleteTags = str(input("Would you like to delete files that contain specific tags? yes/no"))
deleteTags.lower()

if deleteTags == "yes":
    # Prompt the user for input, asking for strings separated by commas
    user_input = input("Please enter tags separated by commas: ")

    # Split the input string into a list based on the comma separator,
    # strip leading/trailing spaces from each element,
    # and filter out any empty strings resulting from a trailing comma.
    input_array = [item.strip() for item in user_input.split(',') if item.strip()]
    print("Array =", input_array)

    # Prompt the user for the folder path containing the text files
    folder_path = image_folder

    # Validate if the folder path exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("Invalid folder path or path does not exist.")
        exit()

    # Initialize an empty list to store names of files with matching content
    matching_files = []

    # Iterate over each file in the directory
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        # Ensure it's a file
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # Open and read the file content with explicit encoding
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Split the content by commas and strip whitespace
                    content_array = [item.strip() for item in content.split(',')]
                    # Check for any matches and add filename to matching_files if found
                    if any(item in input_array for item in content_array):
                        matching_files.append(file_path)
            except UnicodeDecodeError as e:
                print(f"Error reading {filename}: {e}")

    # Print or use the list of matching files as needed
    print("Files with matching content:", matching_files)

    # Function to delete files based on the provided list of file paths
    def delete_files(file_paths):
        for file_path in file_paths:
            # Check if the file exists before attempting deletion
            if os.path.exists(file_path):
                # Delete the text file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
                # Delete image files with the same name as the text file
                base_name = os.path.splitext(file_path)[0]  # Get the base name without extension
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']  # Add more image formats as needed
                for ext in image_extensions:
                    image_file_path = base_name + ext
                    if os.path.exists(image_file_path):
                        os.remove(image_file_path)
                        print(f"Deleted image file: {image_file_path}")

    # Delete the matching files
    delete_files(matching_files)

    # Delete the matching files
    delete_files(matching_files)
else:
    print("Done!")