# CLIP test

This toy web app is leveraging the text-image CLIP model and a Flask web app coded by [ThorkildFregi](https://github.com/ThorkildFregi/CLIP-model-website).

*Dependencies: flask, clip, PIL, torch, numpy*



First, launch:
```
>python3 recurse.py -f static > static_directory.txt
```
if you want to process a folder of images named ``static`` (subfolders may be used within the root folder) stored in the application folder. It will generate the directory files list and a report as text files.

Then the CLIP embeddings are computed with:
```
>python3 model.py -f static
```
The embeddings are saved in a Torch tensor named after the folder name (in our example, ``static_torch.pt``).

Finally, launch the web app:
```
>flask --app main.py --debug run
```
and open this URL http://127.0.0.1:5000 in your browser.

Note: The whole workflow can be run using the bash script ``run.sh``.

The web app displays a random selection of images and a prompt field.

![The web app](screen/home.png)

## Classification scenario

For this use case, we want to use CLIP as a zero-shot classifier. The images types (classes) we want to classify are described in the main.py script as captions, e.g.:
- "a photo"
- "a map"
- "a crossword grid"
...

The results list shows the most likely images for the requested class, the associated probability and the second most likely class. In this example, we are looking for crossword grids in newspapers.

![Classification](screen/classify.png)

## Information retrieval scenario

In this scenario, a free-form query is entered and the result list displays the images ranked in order of probability. In the example below, we are looking for cartoons of people in the street.

![Classification](screen/CBIR.png)


