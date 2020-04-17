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

es_client = Elasticsearch()
bc = BertClient()


async def index_documents(parsed_ids, df, index_name):
    with open("parsed_docs.txt", 'a+') as logfile:
        for _, row in tqdm(df.iterrows()):
            _id = row['cord_uid']
            if _id in parsed_ids:
                continue

            doc = CovidParser.parse(row)
            await asyncio.sleep(0.5)
            sentences = split_single(f"{doc['title']}\n {doc['abstract']}"
                                     .replace('\n', ' '))
            encodings = bc.encode(sentences)
            doc['text_vector'] = np.mean(encodings, axis=1)

            await asyncio.sleep(0.1)
            es_client.index(index=index_name, id=_id, body=doc)
            logfile.write(f'{_id}\n')


def create_es_index(index_file, index_name):
    with open(index_file) as idx_file:
        source = idx_file.read().strip()
        es_client.indices.create(index=index_name, body=source)


@plac.annotations(
    metafile=('positional', 'Path to metadata', None, Path),
    index_config=('positional', 'Mappings for ES Index', None, Path),
    index_name=('positional', 'Index Name', None, Path)
)
def main(metafile: Path = Path('dataset-week1/metadata.csv'),
         index_config: Path = Path('assets/es_config.json'),
         index_name: str = 'covid-april-10'):

    assert metafile.exists()
    assert index_config.exists()

    create_es_index(index_config, index_name)
    df = pd.read_csv(metafile, index_col=None)
    parsed_ids = open('parsed_docs.txt', 'r+').readlines()

    asyncio.run(index_documents(parsed_ids, df, index_name))


if __name__ == '__main__':
    plac.call(main)
