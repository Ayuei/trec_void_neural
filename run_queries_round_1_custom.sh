#!/bin/sh

python3 query.py -query-topics ./assets/topics-rnd1.xml \
                 -index-name "${1}" \
		 -output-file "${2}" \
		 -norm-weight "-1"
		 #-bm25-only
		 #-debug
