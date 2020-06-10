#!/bin/sh

python3 ./es.py -metafile ./datasets/2020-05-19/metadata.csv \
                -index-config ./assets/es_config.json \
                -index-name covid-may-19-fulltext_embed \
		-data-path ./datasets/2020-05-19 \
                -valid-id-path ./assets/docids-rnd3.txt
		#-delete-index
