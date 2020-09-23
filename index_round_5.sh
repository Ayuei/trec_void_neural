#!/bin/sh

module load tensorflow/1.12.0-py37-gpu
module load pytorch/1.4.0-py37-cuda90
module load protobuf/3.6.1
module unload python
module load python/3.7.2
alias vim=vi

#nohup ./elasticsearch-7.6.2/bin/elasticsearch > logs/es_server.nohup 2>&1 &
#nohup bert-serving-start -model_dir models/clinicalcovid-bert-nli/tf_model/ -pooling_layer -1 -pooling_strategy REDUCE_MEAN -port 51234 -port_out 51235 -max_seq_len None -max_batch_size 8 -num_worker=2 > logs/bert_serving.out 2>&1 &

#sleep 240

source venv/bin/activate

python3 -u ./es.py -metafile ./datasets/2020-07-16/metadata.csv \
                -index-config ./assets/es_config.json \
                -index-name covid-round_5 \
		-data-path ./datasets/2020-07-16 \
		-bert-inport 51234 \
		-bert-outport 51235 \
                -valid-id-path ./assets/docids-rnd5.txt #\
		#-delete-index
