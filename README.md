# Elasticsearch Neural Indexing for TREC-COVID

## Requirements
TBA

## How to run
1. Run es.py to index corpus, ensure paths are correct, or correct them flags (see python3 es.py -h)
2. Run query.py to generate TREC-formatted results (see python3 query.py -h)

## Models
ClinicalCovidBERT [https://huggingface.co/manueltonneau/clinicalcovid-bert-nli]
BioBERT STS [https://huggingface.co/clagator/biobert_v1.1_pubmed_nli_sts]
BioBERT msmarco [https://huggingface.co/nboost/pt-biobert-base-msmarco]
CovidBERT [https://huggingface.co/gsarti/covidbert-nli]
