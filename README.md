# CLIP test

This toy web app is leveraging the text-image CLIP model and a Flask web app coded by [ThorkildFregi](https://github.com/ThorkildFregi/CLIP-model-website) (thanks!).

*Dependencies: flask, clip, PIL, torch, numpy*



First, launch:
```
>python3 recurse.py -f static/myImages > static/myImages_directory.txt
```
if you want to process a folder of images named ``myImages`` (subfolders may be used within this folder) stored in the ``static`` application folder. It will generate the directory files list and a report (as both text files).

Then the CLIP embeddings are computed (CUDA or CPU processing is supported) with:
```
>python3 model.py -f myImages
```
The embeddings are saved in a Torch tensor named after the folder name (in our example, ``myImages_torch.pt``).

*Note: ``join.py`` can be used to concatenate tensors if you need to process large volumes of images within memory constraint.*

Then create an env variable to inform the flask app about your images folder:
```
export FLASK_ARG="myImages"
```

Finally, launch the web app:
```
>flask --app main.py --debug run
```
and open this URL http://127.0.0.1:5000 in your browser.

*Note: the whole workflow can be ran using the bash script ``run.sh``. If the image folder is changed, the entire workflow must be restarted.*

The web app displays a random selection of images and a prompt field.

![The web app](screen/home.png)

If the file ``static/myImages_urls.txt`` exists, images are linked to their URL. Images with no URL can be mixed (using the character - as URL).

## Classification scenario

For this use case, we want to use CLIP as a zero-shot classifier. The images types (classes) we want to classify are described in the ``static/myImages_labels.csv`` file as textual captions, e.g.:
```
Crosswords,a crossword grid or a chess game or a word game printed in a heritage newspaper
Drawing,a monochrome drawing printed in a heritage newspaper
Map,a monochrome map printed in a heritage newspaper
Photo,a black and white picture printed in a heritage newspaper
...
```
These class names will be listed to the user on the home page and the model will output probabilities against these captions.

The results list shows the most likely images for the requested class, the probability and the second most likely class. In this example, we are looking for drawings in newspapers. If images are sorted into subfolders based on class names, a confusion matrix will be calculated and displayed after the results list.


![Classification](screen/classify.png)

*Note: if the captions are changed, the flask app must be launched again.* 




## Information retrieval scenario

In this scenario, a free-form query is entered and the result list displays the images ranked in order of probability. In the example below, we are looking for cartoons of people in the street.

![Classification](screen/CBIR.png)


