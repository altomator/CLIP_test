# CLIP test

This toy web app is leveraging the text-image CLIP model and a Flask web app coded by [ThorkildFregi](https://github.com/ThorkildFregi/CLIP-model-website).

*Dependencies: flask, clip, PIL, torch, numpy*



First, launch:
```
>python3 recurse.py -f static > static_directory.txt
```
if you want to process a folder of images named ``static`` (subfolders may be used within the root folder) stored in the application folder. It will generate the directory files list as a text file.

Then the CLIP embeddings are computed with:
```
>python3 model.py -f static
```

Finally, launch the web app:
```
>flask --app main.py --debug run
```
and open this URL http://127.0.0.1:5000 in your browser.

The whole workflow can be run using the bash script ``run.sh``.

## Classification scenario


## Information retrieval scenario

