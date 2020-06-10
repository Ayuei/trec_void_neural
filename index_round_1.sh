#!/bin/sh

python3 ./es.py -metafile ./datasets/covid-april-10/metadata.csv \
                -index-config ./assets/es_config.json \
                -index-name covid-april-10 \
                -valid-id-path ./datasets/covid-april-10/docids-rnd1.txt
		#-delete-index

