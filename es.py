import pandas as pd
from aioelasticsearch import Elasticsearch
from utils.parser import CovidParser as CovidParser
import plac
from pathlib import Path
from tqdm import tqdm
from bert_serving.client import BertClient
import numpy as np
import asyncio
import logging
import traceback
import os
import spacy

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

es_client = Elasticsearch(timeout=600)
bc = BertClient(port=51234, port_out=51235)

async def index_documents(parsed_ids, df, index_name, valid_ids):
    nlp = spacy.load("en_core_sci_sm", disable=['ner', 'tagger'])
    nlp.max_length = 2000000

    with open("parsed_docs.txt", 'a+') as logfile:
        for _, row in tqdm(df.iterrows(), total=len(valid_ids)):
            doc = None
            _id = row['cord_uid'].strip()
            try:
                if not valid_ids[_id] or _id in parsed_ids:
                    logging.warn(f'Skipping {_id} as it\'s already indexed')
                    continue

                # Parse the document fields
                doc = CovidParser.parse(row)
                await asyncio.sleep(0.1)

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
                    sentences = [sent.text.strip() for sent in nlp(doc['abstract']).sents]

                if title_exists:
                    # Batch to save memory and requests
                    sentences.insert(0, doc['title'])

                if sentences:
                    await asyncio.sleep(0.1) # prepare the next batch before we submit our request
                    encodings = bc.encode(sentences)

                if abstract_exists:
                    # if abstract, then title is the first element/vector
                    doc['title_embedding'] = encodings[0].tolist() if title_exists else [0]*768
                else:
                    # If not abstract, the title is the only vector
                    doc['title_embedding'] = encodings.tolist()[0] if title_exists else [0]*768

                if abstract_exists:
                    if title_exists:
                        arr = np.delete(encodings,0,0)
                        #doc['abstract_embedding_array'] = arr.tolist()
                        doc['abstract_embedding'] = np.mean(arr, axis=0).tolist()
                    else:
                        #doc['abstract_embedding_array'] = encodings.tolist()
                        doc['abstract_embedding'] = np.mean(encodings, axis=0).tolist()
                else:
                    # Default to zero vector is no abstract
                    doc['abstract_embedding'] = [0]*768

                if 'fulltext' in doc.keys() and doc['fulltext']:
                    fulltexts = []
                    await asyncio.sleep(0.1)
                    if len(doc['fulltext']) > 2000000:
                        doc['fulltext'] = doc['fulltext'][:2000000]

                    for sent in nlp(doc['fulltext']).sents:
                        text = sent.text.strip()
                        if text:
                            fulltexts.append(text)

                    if fulltexts:
                        full_embedding = bc.encode(fulltexts)
                        if len(full_embedding) > 5000:
                            full_embedding = full_embedding[:5000]
                        #doc['fulltext_embedding_array'] = full_embedding.tolist()
                        doc['fulltext_embedding'] = np.mean(np.array(full_embedding), axis=0).tolist()

                assert len(doc['title_embedding']) == 768
                assert len(doc['abstract_embedding']) == 768

                # Embedding is the bottleneck, so we can perform multiple requests before indexing
                await es_client.index(index=index_name, id=_id, body=doc)
                logfile.write(f'{_id}\n')
            except Exception as e:
                import pdb; pdb.set_trace()
                print(traceback.format_exc())
                logging.critical(f"Cannot process doc {_id}")


async def create_es_index(index_file, index_name, delete=False):
    with open(index_file) as idx_file:
        index_exists = await es_client.indices.exists(index_name)
        if index_exists:
            if delete:
                await es_client.indices.delete(index_name)
                os.remove('parsed_docs.txt')
                open('parsed_docs.txt', 'w+').close() # create empty file
                logging.warn('Deleting old index')
            else:
                return
        source = idx_file.read().strip()
        await es_client.indices.create(index=index_name, body=source)


@plac.annotations(
    metafile=('Path to metadata','option', None, Path),
    index_config=('Mappings for ES Index', 'option', None, Path),
    delete_index=('Delete past index', 'flag', None),
    data_path=("path to the dataset", 'option', None, Path),
    index_name=('Index Name', 'option', None, str),
    valid_id_path=('Path to valid ids', 'option', None, str),
)
def main(metafile: Path = Path('covid-april-10/metadata.csv'),
         index_config: Path = Path('assets/es_config.json'),
         delete_index: bool=False,
         index_name: str = 'covid-april-10',
         data_path: Path= Path("datasets/covid-april-10/"),
         valid_id_path: str = 'covid-april-10/docids-rnd1.txt'):


    assert metafile.exists()
    assert index_config.exists()
    assert data_path.exists()

    loop = asyncio.get_event_loop()
    CovidParser.data_path = str(data_path) + "/"

    loop.run_until_complete(create_es_index(index_config, index_name, delete=delete_index))
    df = pd.read_csv(metafile, index_col=None)

    # Keep a list of parsed documents that we have processed in the event of a crash
    if not os.path.exists('parsed_docs.txt'):
        open('parsed_docs.txt', 'w+') # Create file

    parsed_ids = list(map(lambda k: k.strip(), open('parsed_docs.txt', 'r+').readlines()))
    valid_ids = open(valid_id_path).readlines()

    valid_ids = {_id.strip(): True for _id in open(valid_id_path).readlines()}

    loop.run_until_complete(index_documents(parsed_ids, df, index_name, valid_ids))
    loop.run_until_complete(es_client.transport.close())
    loop.close()


if __name__ == '__main__':
    plac.call(main)
