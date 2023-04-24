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
#from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy

app = Flask(__name__)

# max number of displayed results
maxResults= 100
# max displayed thumbnails on the home page
maxThumbnails = 200

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

# Image files
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
#app.logger.info(image_paths)

# Directory summary
directory_summary = img_folder+ "_directory.txt"
if not os.path.exists(directory_summary):
    app.logger.info("### directory summary does not exist: " + directory_summary)
    quit()
with open(directory_summary) as f:
    summary = f.read()

# Reading the predefined classes for the classification scenario
labels=[]
captions=[]
labels_file = img_folder + "_labels.csv"
if not os.path.exists(labels_file):
    app.logger.info("### file for the class labels is missing: " + labels_file)
    #quit()
else:
    # Building the labels and the GT
    gt_classes_idx=[]
    examples_tot = 0
    paths = ' '.join(image_paths)
    with open(labels_file) as f:
        label = csv.reader(f, delimiter=',')
        for row in label:
            labels.append(row[0])
            captions.append(row[1])
            # we look how many annotated images we have in the subfolder named according to the label
            examples = paths.count(img_folder+"/"+row[0])
            examples_tot += examples
            app.logger.info(" -examples for class "+row[0]+": "+str(examples))
            # we build the array of GT
            gt_idx=[labels.index(row[0])] * examples
            gt_classes_idx.extend(gt_idx)
    classes_nbr=len(labels)
    app.logger.info("   number of images in subfolders: "+ str(examples_tot))
    #app.logger.info(gt_classes_idx)
    conf_mat_file = img_folder+ "_confusion.txt"

def chop(str):
    return str[:5]

def writeConfMat(tensor):
    app.logger.info("writing in : "+conf_mat_file)
    with open(conf_mat_file, 'wb') as conf_file:
        labs = '\t'.join(map(chop,labels))
        app.logger.info(labs)
        conf_file.write('GT\\pred '.encode('utf-8')+labs.encode('utf-8'))
        conf_file.write("\n".encode('utf-8'))
        for i, x in enumerate(tensor.numpy()):
            conf_file.write("\n".encode('utf-8'))
            l = x.tolist()
            values = '\t'.join(map(str,l))
            app.logger.info(values)
            conf_file.write(chop(labels[i]).encode('utf-8')+'\t'.encode('utf-8')+values.encode('utf-8'))

############# MAIN stuff ###############
@app.route('/', methods=("POST", "GET"))
def page():
    if request.method == "POST":
        app.logger.info('...receiving POST request')

        clip.tokenize("Hello world!")
        preprocess

        query = request.form["prompt"]
        if query in labels: ### Classification use case ###
            #minProb = 1.0/classes_nbr # minProb should be > random guess
            #app.logger.info(f"\n-----------------\n threshold prob for classification: {minProb:.2f}")
            label_index = labels.index(query) # the label's index of the targeted class
            caption = captions[label_index] # the caption of the targeted class
            app.logger.info("#################")
            app.logger.info("Query: "+caption)
            app.logger.info(" against "+str(classes_nbr)+" classes")
            text_descriptions = [f"{label}" for label in captions]
            text_tokens = clip.tokenize(text_descriptions)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            #app.logger.info(text_probs)

            # we're looking for the top #2 probs
            top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)
            app.logger.info("####### top probs for the top 2 classes [10 first images] ##########")
            app.logger.info(top_probs[:10])
            app.logger.info(top_labels[:10])
            #app.logger.info(image_paths[:20])

            name_image_top_prob = []
            prob = []
            tuples=[]
            FP=[]
            for i, image in enumerate(image_paths):
                if (top_labels[i][0]==label_index): # filtering the images which match the query
                #filtering the images on a probability threshold for the targeted class
                #if (float(top_probs[i][0]) > minProb and top_labels[i][0]==label_index):
                    # do we have false positives?
                    if query in image_paths[i]:
                        css="green"
                    else:
                        css="crimson"
                    tuples.append((image_paths[i],float(top_probs[i][0]),top_labels[i][1],float(top_probs[i][1]),css))

            app.logger.info("  number of images validating the query: "+ str(len(tuples)))
            app.logger.info(FP)
            # sorting the probs by decreasing values
            decreasing = sorted(tuples, key=lambda criteria: criteria[1], reverse=True)
            # build the arrays for the HTML rendition: maxResults first items
            name_image_top_prob = [i[0] for i in decreasing[:maxResults]]
            #app.logger.info("####### files for the result images ##########")
            #app.logger.info(name_image_top_prob)
            prob1 = [i[1] for i in decreasing[:maxResults]]
            string_prob1 = ["%.2f" % number for number in prob1]
            FP=[i[4] for i in decreasing[:maxResults]]
            app.logger.info("####### top prob #1 for the targeted class ##########")
            app.logger.info(string_prob1)

            # the second best class
            prob2 = [i[3] for i in decreasing[:maxResults]]
            string_prob2 = ["%.2f" % number for number in prob2]
            class2 = [labels[i[2]] for i in decreasing[:maxResults]]
            app.logger.info("####### top prob #2 for the result images ##########")
            app.logger.info(string_prob2)
            app.logger.info(class2)

            # computing the confusion matrix
            # https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
            if image_count != examples_tot:
                conf_msg="## Confusion Matrix calculation: something is wrong with the ground truth!\n(images must be stored in subfolders; subfolders must be named accordingly to labels; labels must be sorted) ##"
                app.logger.info(conf_msg)
                app.logger.info("images files: "+str(image_count))
                app.logger.info("images in GT folders: "+str(examples_tot))
                return render_template("grid_classif.html", files=name_image_top_prob, labels= ' / '.join(labels), prob1=string_prob1, targetClass=query, class2=class2, prob2=string_prob2, query=query, caption=caption, confmatfile="static/_confusion.txt", accuracy1="", accuracy2="", comment="("+str(len(tuples))+ " results, first "+str(maxResults)+" displayed)")
            else:
                app.logger.info("...computing the confusion matrix")
                pred_classes_idx = torch.argmax(text_probs, dim=1)
                #app.logger.info(pred_classes_idx)
                #pred_class_names = [labels[i] for i in pred_class_idx]
                pred = torch.tensor(pred_classes_idx)
                target = torch.tensor(gt_classes_idx)
                #app.logger.info(target)
                conf_mat = MulticlassConfusionMatrix(num_classes=classes_nbr)
                confusion = conf_mat(pred,target)
                # write the matrix in a txt file
                writeConfMat(confusion)
                # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                metric = MulticlassAccuracy(num_classes=classes_nbr)
                acc = metric(pred,target) * 100.0
                accuracy = "%.2f" % acc
                app.logger.info("accuracy (mean of classes): "+ accuracy)
                metric2 = MulticlassAccuracy(num_classes=classes_nbr, average=None)
                acc = metric2(pred,target)
                app.logger.info(acc)
                acc_class  = [i * 100.0 for i in acc]
                acc_class = ["%.2f" % number for number in acc_class]
                string_acc = ' %  /  '.join(acc_class)
                return render_template("grid_classif.html", files=name_image_top_prob, labels= ' / '.join(labels), targetClass=query, caption=caption, prob1=string_prob1, fp=FP, class2=class2, prob2=string_prob2,  confmatfile=conf_mat_file, accuracy1=accuracy, accuracy2=string_acc, comment="("+str(len(tuples))+ " results, first "+str(maxResults)+" displayed)")
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
            name_image_top_prob = [image_paths[i] for i in indices[:maxResults]]
            #app.logger.info(name_image_top_prob)
            prob = sorted_values[:maxResults]
            stringProb = ["%.3f" % number for number in prob]
            return render_template("grid_cbir.html", files=name_image_top_prob, prob=stringProb, query=query, comment="(first "+str(maxResults)+" results displayed)")
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
        app.logger.info(labels)
        if not labels:
            return render_template("home.html", files=random_files, msg='Use a free query:', classes='', count=image_count, summary=summary)
        else:
            return render_template("home.html", files=random_files, msg='Use a free query OR one of these predefined classes: ', classes=', '.join(labels), count=image_count, summary=summary)
