#!/bin/bash
# bash script

export FLASK_ARG="dsc"

echo "-------------------------"
if [ -z "${FLASK_ARG}" ]
then
  echo "# FLASK_ARG env var must be defined and must target a folder of images in static/ #"
  exit 0
fi

echo "FLASK_ARG env var is set to: ${FLASK_ARG}"

if [ ! -f static/${FLASK_ARG}.txt ]
then
    echo "... listing the directory ${FLASK_ARG}"
    python3 recurse.py -f static/$FLASK_ARG > static/${FLASK_ARG}_directory.txt
fi

if [ ! -f ${FLASK_ARG}_torch.pt ]
then
    echo "... generating the embeddings for static/${FLASK_ARG}"
    python3 model.py -f ${FLASK_ARG}
fi
echo "... lauching the web app"
flask --app main.py --debug run
