# Generates the CLIP embeddings for a list of image files
# output: Torch tensor
# input: folder name
# prerequisite: directory list of the target folder

import numpy as np
import torch # python3 -m pip install torch
import os
from PIL import Image
import clip # python3 -m pip install clip git+https://github.com/openai/CLIP.git
import pathlib
import argparse

###
parser = argparse.ArgumentParser(
                    prog = 'model.py',
                    description = 'Generates the CLIP embeddings for a list of files'
                    )
parser.add_argument('-f', '-folderName', help='the folder to be listed',required=True)
args = parser.parse_args()
img_folder = args.f

if not os.path.exists(img_folder):
    print("### '%s' folder does not exist! ###\n" % img_folder)
    quit()

print("...reading directory for folder: ", img_folder)
directory_file = img_folder+ ".txt"
if not os.path.exists(directory_file):
    print("### directory list for folder %s does not exist! ###\nPlease run \n>python recurse.py -f %s \nfirst" % (directory_file,img_folder))
    quit()

with open(directory_file) as f:
    image_paths = f.readlines()
image_count = len(image_paths)
print("   number of images found:", image_count)

# CLIP
print("...loading the CLIP model")
print("   Torch version:", torch.__version__)

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("   model parameters: " f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("   input resolution:", input_resolution)
print("   context length:", context_length)
print("   vocab size:", vocab_size)

# Image Preprocessing
# The second return value from clip.load() contains a torchvision Transform that performs this preprocessing.
preprocess

# text preprocessing
clip.tokenize("Hello world!")

images = []
i = 0
print("...processing the images")

for filename in image_paths:
    #name = os.path.splitext(filename)[0]
    filename = filename[:-1]
    if (i % 10 == 0):
        print(i, " : ", filename)
    image = Image.open(filename).convert("RGB")
    images.append(preprocess(image))
    i += 1

print ("...buiding the torch tensor")
image_input = torch.tensor(np.stack(images))

print ("...now generating the image features")
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

output = img_folder + "_torch.pt"
print ("--> embeddings saved in %s\n" % output)
torch.save(image_features, output)
