# > flask --app main.py --debug run
# puis http://127.0.0.1:5000

#import logging
from flask import Flask, redirect, request, url_for, render_template
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
from torchmetrics.classification import MulticlassRecall

app = Flask(__name__)

# max number of displayed results
maxResults= 300
# max displayed thumbnails on the home page
maxThumbnails = 150
# folder where the tensors are
torchFolder = "torch/"
# thresholds
minProb=0.6
gap=0.8 # between top 1 and top 2 for top2 inferences

# get the targeted folder
try:
    use_case = os.environ["FLASK_ARG"]
except:
    print("### FLASK_ARG env var is not defined! ###")
    quit()

# app folders
img_folder = "static/"+use_case
eval_folder = "static/_eval/"
conf_mat_file = eval_folder+use_case+"_confusion.txt"

app.logger.info('...working on image folder: '+img_folder)

app.logger.info('...loading the CLIP model')
print(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# Tensor
tensor_file = torchFolder + use_case + "_torch.pt"
if not os.path.exists(tensor_file):
    app.logger.info("### Torch tensor is missing: "+tensor_file)
    quit()
app.logger.info("...loading the embeddings from: "+tensor_file)
image_features = torch.load(tensor_file, map_location=torch.device('cpu'))
app.logger.info(image_features.size())

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

if image_count != image_features.size()[0]:
    app.logger.info("### Tensor size and images number are different!")
    quit()

# Directory summary
directory_summary = img_folder+ "_directory.txt"
if not os.path.exists(directory_summary):
    app.logger.info("### directory summary does not exist: " + directory_summary)
    quit()
with open(directory_summary) as f:
    summary = f.read()

# URLs
urls=[]
image_urls = img_folder+ "_urls.txt"
if not os.path.exists(image_urls):
    app.logger.info("### URLs list does not exist: " + image_urls)
else:
    app.logger.info('...reading the URLs list')
    with open(image_urls) as f:
        urls = f.readlines()
    #app.logger.info(urls)
    urls_count = len(urls)
    app.logger.info("   number of URLs found: "+ str(urls_count))

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
            examples = paths.count(use_case+"/"+row[0])
            examples_tot += examples
            app.logger.info(" -examples for class "+row[0]+": "+str(examples))
            # we build the array of GT
            gt_idx=[labels.index(row[0])] * examples
            gt_classes_idx.extend(gt_idx)
    classes_nbr=len(labels)
    app.logger.info("   number of images in subfolders: "+ str(examples_tot))
    #app.logger.info(gt_classes_idx)

def chop(str):
    return str[:5]

def writeConfMat(tensor, filename, accuracy1, accuracy2):
    app.logger.info("writing in : "+filename)
    with open(filename, 'wb') as conf_file:
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
        conf_file.write("\n\n".encode('utf-8'))
        conf_file.write("Accuracy (micro average): ".encode('utf-8')+ accuracy1.encode('utf-8'))
        conf_file.write("\n".encode('utf-8'))
        conf_file.write("Accuracy (per classe): \n".encode('utf-8'))
        heading= ' / '.join(labels)
        conf_file.write(heading.encode('utf-8'))
        conf_file.write("\n".encode('utf-8'))
        conf_file.write(accuracy2.encode('utf-8'))

def writeTensor(tensor, filename):
    if os.path.exists(filename):
        app.logger.info(filename+" already exists!")
        return
    #app.logger.info(tensor)
    app.logger.info("writing in : "+filename)
    with open(filename, 'wb') as conf_file:
        for i, x in enumerate(tensor.numpy()):
            l = x.tolist()
            if type(l) is list:
                values = ';'.join(map(str,l))
            else:
                values = str(l)
            #app.logger.info(values)
            conf_file.write(values.encode('utf-8'))
            conf_file.write("\n".encode('utf-8'))

def writeList(a_list, filename):
    if os.path.exists(filename):
        app.logger.info(filename+" already exists!")
        return
    #app.logger.info(a_list)
    app.logger.info("writing in : "+filename)
    with open(filename, 'wb') as conf_file:
        for x in a_list:
            value = str(x)
            #app.logger.info(values)
            conf_file.write(value.encode('utf-8'))
            conf_file.write("\n".encode('utf-8'))

############# MAIN stuff ###############
@app.route('/', methods=("POST", "GET"))
def page():
    if request.method == "POST":
        app.logger.info('...receiving POST request')

        clip.tokenize("Hello world!")
        preprocess

        query = request.form["prompt"]

        ### Classification use case ###
        if query in labels:
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
            writeTensor(top_probs, eval_folder+use_case+"_probs2.txt")
            app.logger.info("####### top probs for the top-2 classes [10 first images] ##########")
            app.logger.info(top_probs[:10])
            app.logger.info("####### label index for the top-2 classes")
            app.logger.info(top_labels[:10])
            #app.logger.info(image_paths[:20])

            name_image_top_prob = []
            urls_top_prob = []
            prob = []
            tuples=[]
            for i, image in enumerate(image_paths):
                # filtering the images which match the query
                # top 1 prob or top 2 prob criteria
                if ((top_labels[i][0]==label_index) or (top_labels[i][1]==label_index and top_probs[i][1]/top_probs[i][0] > gap)):
                # top 1 prob only
                #if (top_labels[i][0]==label_index) :
                #filtering the images on a probability threshold for the targeted class
                #if (float(top_probs[i][0]) > minProb and top_labels[i][0]==label_index):
                    # do we have false positives?
                    if query in image_paths[i]:
                        if (top_labels[i][0]==label_index):
                            css1="green"
                            css2="silver"
                        else:
                            css1="silver"
                            css2="green"
                    else:
                        css1="crimson"
                        css2="crimson"
                    tuples.append((image_paths[i],top_labels[i][0],float(top_probs[i][0]),top_labels[i][1],float(top_probs[i][1]),css1,css2,i))

            app.logger.info("  number of images validating the query: "+ str(len(tuples)))

            # sorting the probs by decreasing values
            decreasing = sorted(tuples, key=lambda criteria: criteria[2], reverse=True)
            # build the arrays for the HTML rendition: maxResults first items
            name_image_top_prob = [i[0] for i in decreasing[:maxResults]]
            FP1=[i[5] for i in decreasing[:maxResults]]
            FP2=[i[6] for i in decreasing[:maxResults]]
            if len(urls) != 0:
                urls_top_prob = [i[7] for i in decreasing[:maxResults]]
            app.logger.info("####### files for the result images ##########")
            app.logger.info(name_image_top_prob[0:10])
            prob1 = [i[2] for i in decreasing[:maxResults]]
            string_prob1 = ["%.2f" % number for number in prob1]
            class1 = [labels[i[1]] for i in decreasing[:maxResults]]
            app.logger.info("####### top prob #1 for the result images ##########")
            app.logger.info(string_prob1[0:10])
            app.logger.info("####### class of the top prob #1 ##########")
            app.logger.info(class1[0:10])

            # the second best class
            prob2 = [i[4] for i in decreasing[:maxResults]]
            string_prob2 = ["%.2f" % number for number in prob2]
            class2 = [labels[i[3]] for i in decreasing[:maxResults]]
            app.logger.info("####### top prob #2 for the result images ##########")
            app.logger.info(string_prob2[0:10])
            app.logger.info("####### class of the top prob #2 ##########")
            app.logger.info(class2[0:10])

            # computing the confusion matrix
            # https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
            if os.path.exists(conf_mat_file):
                app.logger.info(conf_mat_file+" already exists!")
                confmatfile=conf_mat_file
            elif image_count != examples_tot:
                conf_msg="## Confusion Matrix calculation: something is wrong with the ground truth!\n(images must be stored in subfolders; subfolders must be named accordingly to labels; labels must be sorted) ##"
                app.logger.info(conf_msg)
                app.logger.info("images files: "+str(image_count))
                app.logger.info("images in GT folders: "+str(examples_tot))
                confmatfile="static/_eval/_confusion.txt"
                #return render_template("grid_classif.html", files=name_image_top_prob, urls=urls_top_prob, target_class=query, use_case=use_case, caption=caption, class1=class1, prob1=string_prob1, fp1=[], fp2=[], class2=class2, prob2=string_prob2, confmatfile="static/_confusion.txt", comment="("+str(len(tuples))+ " results, first "+str(maxResults)+" displayed)")
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
                # write the evaluation data
                writeTensor(pred, eval_folder+use_case+"_pred.txt")
                writeTensor(target, eval_folder+use_case+"_GT.txt")
                GT_labels = [labels[i] for i in target]
                writeList(GT_labels, eval_folder+use_case+"_GT_labels.txt")
                pred_labels = [labels[i] for i in pred]
                writeList(pred_labels, eval_folder+use_case+"_pred_labels.txt")

                # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                metric = MulticlassAccuracy(num_classes=classes_nbr, average="micro")
                acc = metric(pred,target) * 100.0
                accuracy1 = "%.2f" % acc
                app.logger.info(accuracy1)
                metric = MulticlassAccuracy(num_classes=classes_nbr, average=None)
                acc = metric(pred,target)
                acc_class  = [i * 100.0 for i in acc]
                acc_class = ["%.2f" % number for number in acc_class]
                accuracy2 = ' % / '.join(acc_class)+" %"
                app.logger.info(accuracy2)
                # write the matrix in a txt file
                writeConfMat(confusion, conf_mat_file, accuracy1, accuracy2)
                confmatfile=conf_mat_file
                #metric = MulticlassRecall(num_classes=classes_nbr, average=none)
                #rec = metric(pred,target) * 100.0
                #recall = "%.2f" % rec
                #app.logger.info("Recall (mean of classes): "+ recall)
            return render_template("grid_classif.html", files=name_image_top_prob, urls=urls_top_prob, nlabels=classes_nbr, target_class=query, use_case=use_case, caption=caption, class1=class1, prob1=string_prob1, fp1=FP1, fp2=FP2, class2=class2, prob2=string_prob2, confmatfile=confmatfile, comment="("+str(len(tuples))+ " results, first "+str(maxResults)+" displayed)")
        else:
            ### Image retrieval use case ###
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
            sorted_values, indices = torch.sort(text_probs, dim=0, descending=True)
            name_image_top_prob = [image_paths[i] for i in indices[:maxResults]]
            urls_top_prob = []
            if len(urls) != 0:
                urls_top_prob = [urls[i] for i in indices[:maxResults]]
            #app.logger.info(name_image_top_prob)
            prob = sorted_values[:maxResults]
            stringProb = ["%.3f" % number for number in prob]
            return render_template("grid_cbir.html", files=name_image_top_prob, urls=urls_top_prob, prob=stringProb, query=query, use_case=use_case, comment="(first "+str(maxResults)+" results displayed)")
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
            return render_template("home.html", files=random_files, urls=urls, msg='Use a free query:', classes='', count=image_count, summary=summary)
        else:
            return render_template("home.html", files=random_files, urls=urls, msg='Use a free query OR one of these predefined classes: ', classes=', '.join(labels), count=image_count, summary=summary)
