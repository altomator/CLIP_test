# > flask --app main.py --debug run
# puis http://127.0.0.1:5000

import logging
from flask import Flask, redirect, request, url_for, render_template
#import click
from PIL import Image
from tqdm import tqdm
import torch
import os
import clip
import pathlib
import random
import numpy as np
import csv

app = Flask(__name__)

# get the targeted folder
try:
    use_case = os.environ["FLASK_ARG"]
except:
    print("### FLASK_ARG env var is not defined! ###")
    quit()

img_folder = "static/"+use_case

app.logger.info('...working on image folder: '+img_folder)

app.logger.info('...loading the CLIP model')
model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# Tensor
tensor_file = use_case + "_torch.pt"
if not os.path.exists(tensor_file):
    app.logger.info("### Torch tensor is missing: "+tensor_file)
    quit()
app.logger.info("...loading the embeddings from: "+tensor_file)
image_features = torch.load(tensor_file)

# Files
if not os.path.exists(img_folder):
    app.logger.info("### images folder does not exist: "+ img_folder)
    quit()
directory_file = img_folder+ ".txt"
app.logger.info('...reading the directory list: '+directory_file)
if not os.path.exists(directory_file):
    app.logger.info("### directory list does not exist: " + directory_file)
    quit()
with open(directory_file) as f:
    image_paths = f.readlines()
image_count = len(image_paths)
app.logger.info("   number of images found: "+ str(image_count))

# Directory summary
directory_summary = img_folder+ "_directory.txt"
if not os.path.exists(directory_summary):
    app.logger.info("### directory summary does not exist: " + directory_summary)
    quit()
with open(directory_summary) as f:
    summary = f.read()

# max number of displayed results
maxResults= 100
# max displayed thumbnails on the home page
maxThumbnails = 200

# predefined classes for the classification scenario
labels=[]
captions=[]
labels_file = img_folder+ "_labels.csv"
if not os.path.exists(labels_file):
    app.logger.info("### CSV labels file is missing: " + labels_file)
    quit()
with open(labels_file) as f:
    label = csv.reader(f, delimiter=',')
    for row in label:
        labels.append(row[0])
        captions.append(row[1])
app.logger.info(labels)

############# MAIN stuff ###############
@app.route('/', methods=("POST", "GET"))
def page():
    if request.method == "POST":
        app.logger.info('...receiving POST request')

        clip.tokenize("Hello world!")
        preprocess

        query = request.form["prompt"]
        if query in labels: ### Classification use case ###
            minProb = 1.0/len(labels) # minProb should be > random guess
            app.logger.info(f"\n-----------------\n threshold prob for classification: {minProb:.2f}")
            label_index = labels.index(query) # the label's index of the targeted class
            caption = captions[label_index] # the caption of the targeted class
            app.logger.info("#################")
            app.logger.info(caption)
            text_descriptions = [f"{label}" for label in captions]
            text_tokens = clip.tokenize(text_descriptions)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # we're looking for the top #2 probs
            top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)
            app.logger.info("####### top probs for the top 2 classes [10] ##########")
            app.logger.info(top_probs[:10])
            app.logger.info(top_labels[:10])
            #app.logger.info(image_paths[:20])

            nameImageTopProb = []
            prob = []
            tuples=[]
            # filtering the images on a probability threshold for the targeted class
            for i, image in enumerate(image_paths):
                if (top_labels[i][0]==label_index):
                #if (float(top_probs[i][0]) > minProb and top_labels[i][0]==label_index):
                    tuples.append((image_paths[i],float(top_probs[i][0]),top_labels[i][1],float(top_probs[i][1])))
            app.logger.info("  number of images validating the query: "+ str(len(tuples)))
            # sorting the probs by decreasing values
            decreasing = sorted(tuples, key=lambda criteria: criteria[1], reverse=True)
            # build the arrays for the HTML rendition: maxResults first items
            nameImageTopProb = [i[0] for i in decreasing[:maxResults]]
            app.logger.info("####### files for the result images ##########")
            app.logger.info(nameImageTopProb)
            prob1 = [i[1] for i in decreasing[:maxResults]]
            stringProb1 = ["%.3f" % number for number in prob1]
            app.logger.info("####### top prob #1 for the targeted class ##########")
            app.logger.info(stringProb1)
            # the second best class
            prob2 = [i[3] for i in decreasing[:maxResults]]
            stringProb2 = ["%.3f" % number for number in prob2]
            class2 = [labels[i[2]] for i in decreasing[:maxResults]]
            app.logger.info("####### top prob #2 for the result images ##########")
            app.logger.info(stringProb2)
            app.logger.info(class2)
            return render_template("grid_classif.html", files=nameImageTopProb, prob1=stringProb1, targetClass=query, class2=class2, prob2=stringProb2, query=query, caption=caption, comment="("+str(len(tuples))+ " results, first "+str(maxResults)+" displayed)")
        else:
            ### Image retrieval use case ###
            #text_descriptions=["a photo of a blank page"]
            text_descriptions=[]
            query = request.form["prompt"]
            text_descriptions.append(f""+query)
            app.logger.info(text_descriptions)
            text_tokens = clip.tokenize(text_descriptions)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (image_features @ text_features.T)
            #app.logger.info(text_probs)

            sorted_values, indices = torch.sort(text_probs, dim=0,descending=True)
            nameImageTopProb = [image_paths[i] for i in indices[:maxResults]]
            app.logger.info(nameImageTopProb)
            prob = sorted_values[:maxResults]
            stringProb = ["%.3f" % number for number in prob]
            return render_template("grid_cbir.html", files=nameImageTopProb, prob=stringProb, query=query, comment="(first "+str(maxResults)+" results displayed)")
    else:
        # Building the home page: images mosaic
        app.logger.info('...Starting to populate the image grid')
        app.logger.info('   from folder: '+ img_folder)

        random_files = []
        i = 0
        n = 0
        r = int(image_count/maxThumbnails)
        for filename in image_paths:
            # we want to randomly select 'maxThumbnails' pictures on the whole dataset
            n+=1
            rdm = random.randint(0, r)
            if rdm == r:
                filename = filename[:-1] # chop the last char = return
                random_files.append(filename)
                i+=1
            if i >= maxThumbnails:
                break
        app.logger.info(f"   files: {n}")
        return render_template("home.html", files=random_files, classes=', '.join(labels), count=image_count, summary=summary)
