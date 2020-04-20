import pandas as pd
from elasticsearch import Elasticsearch
from utils.parser import CovidParser
import plac
from pathlib import Path
from tqdm import tqdm
from segtok.segmenter import split_single
from bert_serving.client import BertClient
import numpy as np
import asyncio
import logging
import traceback

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

es_client = Elasticsearch()
bc = BertClient(port=51234, port_out=51235)


async def index_documents(parsed_ids, df, index_name, valid_ids):
    with open("parsed_docs.txt", 'a+') as logfile:
        for _, row in tqdm(df.iterrows(), total=len(valid_ids)):
            doc = None
            try:
                _id = row['cord_uid'].strip()
                if not valid_ids[_id] or _id in parsed_ids:
                    logging.warn(f'Skipping {_id} as it\'s already indexed')
                    continue

                # Parse the document fields
                doc = CovidParser.parse(row)
                await asyncio.sleep(0.5)

                title_exists = True
                if pd.isnull(doc['title']): # Check if a title exists
                    logging.warn('{_id} has no title')
                    doc['title'] = ''
                    title_exists = False

                # Check if abstract exists before performing code that requires it to be non-empty
                abstract_exists = True
                if doc['abstract'].lower() == 'Unknown' or len(doc['abstract'].strip()) == 0:
                    logging.warn('{_id} has no abstract')
                    abstract_exists = False

                sentences = []

                if abstract_exists:
                    sentences = split_single(doc['abstract'])

                if title_exists:
                    # Batch to save memory and requests
                    sentences.insert(0, doc['title'])

                if sentences:
                    encodings = bc.encode(sentences)

                if abstract_exists:
                    # if abstract, then title is the first element/vector
                    doc['title_embedding'] = encodings[0].tolist() if title_exists else [0]*768
                else:
                    # If not abstract, the title is the only vector
                    doc['title_embedding'] = encodings.tolist()[0] if title_exists else [0]*768

                if abstract_exists:
                    if title_exists:
                        doc['abstract_embedding'] = np.mean(np.delete(encodings,0,0), axis=0).tolist()
                    else:
                        doc['abstract_embedding'] = np.mean(encodings, axis=0).tolist()
                else:
                    # Default to zero vector is no abstract
                    doc['abstract_embedding'] = [0]*768

                assert len(doc['title_embedding']) == 768
                assert len(doc['abstract_embedding']) == 768

                # Embedding is the bottleneck, so we can perform multiple requests before indexing
                await asyncio.sleep(0.1)
                es_client.index(index=index_name, id=_id, body=doc)
                logfile.write(f'{_id}\n')
            except Exception as e:
                logging.critical(traceback.format_exc())
                logging.critical(f'Some unknown error occured, skipping {_id} and continuing')


def create_es_index(index_file, index_name, delete=False):
    with open(index_file) as idx_file:
        if es_client.indices.exists(index_name):
            if delete:
                es_client.indices.delete(index_name)
                import os
                os.remove('parsed_docs.txt')
                open('parsed_docs.txt', 'w+').close() # create empty file
                logging.warn('Deleting old index')
            else:
                return
        source = idx_file.read().strip()
        es_client.indices.create(index=index_name, body=source)


@plac.annotations(
    metafile=('Path to metadata','positional', None, Path),
    index_config=('Mappings for ES Index', 'positional', None, Path),
    delete_index=('Delete past index', 'flag', None),
    index_name=('Index Name', 'positional', None, Path)
)
def main(metafile: Path = Path('covid-april-10/metadata.csv'),
         index_config: Path = Path('assets/es_config.json'),
         delete_index: bool=False,
         index_name: str = 'covid-april-10',
         valid_id_path: str = 'covid-april-10/docids-rnd1.txt'):

    assert metafile.exists()
    assert index_config.exists()

    create_es_index(index_config, index_name, delete=delete_index)
    df = pd.read_csv(metafile, index_col=None)

    # Keep a list of parsed documents that we have processed in the event of a crash
    parsed_ids = list(map(lambda k: k.strip(), open('parsed_docs.txt', 'r+').readlines()))
    valid_ids = open(valid_id_path).readlines()

    valid_ids = {_id.strip(): True for _id in open(valid_id_path).readlines()}
    asyncio.run(index_documents(parsed_ids, df, index_name, valid_ids))


if __name__ == '__main__':
    plac.call(main)
