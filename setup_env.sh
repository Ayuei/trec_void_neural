#!/bin/sh

source venv/bin/activate

module load tensorflow/1.12.0-py37-gpu
module load pytorch/1.4.0-py37-cuda90
module load protobuf/3.6.1
alias vim=vi


nohup bert-serving-start -model_dir biobert_nli/model_tf/ -pooling_layer -1 -pooling_strategy CLS_TOKEN -port 51234 -port_out 51235 -max_seq_len None > logs/bert_serving.nohup 2>&1 &
nohup ./elasticsearch-7.6.2/bin/elasticsearch > logs/es_server.nohup 2>&1 &
