#!/bin/sh

set -e

#weights=(1.50 1.75 2.00 2.15 2.20 2.40 2.60 3.00 5.00)
weights=("-1")

for weight in ${weights[@]}; do
	file_out="temptemp"
	echo $file_out 
	python3 query.py -query-topics ./assets/new_topics.pkl \
                 -index-name covid-round_5 \
		 -norm-weight "${weight}" \
		 -output-file "${file_out}" \
		 -exclude narrative \

	./trec_eval -m ndcg_cut.10 ./assets/qrels-covid_d4_j0.5-4.txt "${file_out}"
	./trec_eval assets/qrels-covid_d4_j0.5-4.txt "${file_out}"
	#rm "${file_out}"
done
