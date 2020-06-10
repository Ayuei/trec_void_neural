#!/bin/sh

module load tensorflow/1.12.0-py37-gpu
module load pytorch/1.4.0-py37-cuda90
module load protobuf/3.6.1
module unload python
module load python/3.7.2
alias vim=vi

nohup bert-serving-start -model_dir biobert_nli/model_tf/ -pooling_layer -1 -pooling_strategy REDUCE_MEAN -port 51234 -port_out 51235 -max_seq_len None -max_batch_size 1 -num_worker=4 > logs/bert_serving.nohup 2>&1 &
nohup ./elasticsearch-7.6.2/bin/elasticsearch > logs/es_server.nohup 2>&1 &

source venv/bin/activate
