#!/bin/sh

infile=${1}
trecfile=${2}
#./trec_eval -m ndcg_cut.10 ./assets/qrels-covid_d5_j0.5-5.txt "${infile}"
#./trec_eval assets/qrels-covid_d5_j0.5-5.txt "${infile}"
#./trec_eval -m ndcg_cut.10 ./assets/qrels-covid_d3_j0.5-3.txt  "${infile}"
#./trec_eval assets/qrels-covid_d3_j0.5-3.txt "${infile}"
./trec_eval -m ndcg_cut.10 "${trecfile}" "${infile}"
./trec_eval "${trecfile}" "${infile}"
