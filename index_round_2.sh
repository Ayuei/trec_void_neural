#!/bin/sh

python3 ./es.py -metafile ./datasets/covid-may-1/metadata.csv \
                -index-config ./assets/es_config.json \
                -index-name covid-may-1-fulltext_embed \
		-data-path ./datasets/covid-may-1/ \
                -valid-id-path ./datasets/covid-may-1/docids-rnd2.txt
		#-delete-index
