#!/bin/sh

set -e

#weights=(1.50 1.75 2.00 2.15 2.20 2.40 2.60 3.00 5.00)
weights=(2.50)

for weight in ${weights[@]}; do
	file_out="${weight}_comb_remove_divisor_check.txt"
	echo $file_out 
	python3 query.py -query-topics ./assets/topics-rnd4.xml \
                 -index-name covid-round_4 \
		 -norm-weight "${weight}" \
		 -output-file "${file_out}"

	./trec_eval -m ndcg_cut.10 assets/qrels-covid_d3_j0.5-3.txt "${file_out}"
	./trec_eval assets/qrels-covid_d3_j0.5-3.txt "${file_out}"
	#rm "${file_out}"
done
