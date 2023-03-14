#!/bin/bash
# bash script

export FLASK_ARG="static"
echo "... listing the directory"
python3 recurse.py -f static > static_directory.txt
echo "... generating the embeddings"
python3 model.py -f static
echo "...lauching the web app"
flask --app main.py --debug run
