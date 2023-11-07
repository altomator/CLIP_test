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

# folder where the tensors are
torchFolder = "torch/"

###
parser = argparse.ArgumentParser(
                    prog = 'model.py',
                    description = 'Generates the CLIP embeddings for a list of images'
                    )
parser.add_argument('-f', '-folderName', help='the folder to be processed (must be in static/)',required=True)
args = parser.parse_args()
img_folder = "static/"+args.f

if not os.path.exists(img_folder):
    print("### '%s' folder does not exist! ###\n" % img_folder)
    quit()

print("...reading directory list for: ", img_folder)
directory_file = img_folder + ".txt"
if not os.path.exists(directory_file):
    print("### directory list for %s does not exist! ###\nPlease run \n>python recurse.py -f %s \nfirst" % (directory_file,img_folder))
    quit()

with open(directory_file) as f:
    image_paths = f.readlines()
image_count = len(image_paths)
print("   number of images found:", image_count)

# CLIP
print("...loading the CLIP model")
print("   Torch version:", torch.__version__)

print(clip.available_models())

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32",device=device)
model.to(device).eval()

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
clip.tokenize("Hello world!").to(device)

images = []
i = 0
print("...processing the images")

for filename in image_paths:
    #name = os.path.splitext(filename)[0]
    filename = filename[:-1] # chop the last char = return
    if (i % 10 == 0):
        print(i, " : ", filename)
    image = Image.open(filename).convert("RGB")
    #image = preprocess(Image.open(filename).convert("RGB")).unsqueeze(0).to(device)
    images.append(preprocess(image))
    #images.append(image)
    i += 1

print ("...building the torch tensor")
image_input = torch.tensor(np.stack(images)).to(device)


print ("...now generating the image features")
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

#
output = torchFolder + args.f +  "_torch.pt"

print ("--> embeddings saved in %s\n" % output)
torch.save(image_features, output)
