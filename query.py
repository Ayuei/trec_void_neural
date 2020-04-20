import plac
from aioelasticsearch import Elasticsearch
import asyncio
from xml.etree import ElementTree as ET
from pathlib import Path
from bert_serving.client import BertClient
import logging
from tqdm import tqdm
import numpy as np

# Final score = log(BM25 of query/question/narrative) + cosine(query, doc_title) + cosine(question, doc_title) + cosine(narr, doc_title)
# + cosine(query, doc_abstract) + cosine(question, doc_abstract) + cosine(narr, doc_abstract)
# Log on bm25 is used as it generally gives a score from 0 - 6 which is the range of our cosine similarity
def generate_query(q, qstn, narr, q_eb, qstn_eb, narr_eb):
    return {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        # Match on title, abstract and fulltext on all three fields
                        # Weights should be added later
                        "should": [
                            {"match": {"title": q}},
                            {"match": {"title": qstn}},
                            {"match": {"title": narr}},
                            {"match": {"abstract": q}},
                            {"match": {"abstract": qstn}},
                            {"match": {"abstract": narr}},
                            {"match": {"fulltext": q}},
                            {"match": {"fulltext": qstn}},
                            {"match": {"fulltext": narr}},
                        ]
                    }
                },
                "script":{
                    "lang": "painless",
                    # Compute dotproducts as some Vectors are zero vectors
                    # Use dotproducts as a proxy to see if we're able to compute the cosine similarity
                    # Otherwise return 0
                    # We have to do this as the values of Vectors in elasticsearch are not only
                    # PRIVATE but ALSO encoded in BINARY that non-trivally decoded.
                    # Weights should be added later
                    "source": """
                               double q_t = dotProduct(params.q_eb, 'title_embedding');
                               double qstn_t = dotProduct(params.qstn_eb, 'title_embedding');
                               double narr_t = dotProduct(params.narr_eb, 'title_embedding');
                               double q_abs = dotProduct(params.q_eb, 'abstract_embedding');
                               double qstn_abs = dotProduct(params.qstn_eb, 'abstract_embedding');
                               double narr_abs = dotProduct(params.narr_eb, 'abstract_embedding');

                               if (Math.signum(q_t) != 0){
                                   q_t = cosineSimilarity(params.q_eb, 'title_embedding');
                               }

                               if (Math.signum(qstn_t) != 0){
                                   qstn_t = cosineSimilarity(params.qstn_eb, 'title_embedding');
                               }

                               if (Math.signum(narr_t) != 0){
                                   narr_t = cosineSimilarity(params.narr_eb, 'title_embedding');
                               }

                               if (Math.signum(q_abs) != 0){
                                   q_abs = cosineSimilarity(params.q_eb, 'abstract_embedding');
                               }

                               if (Math.signum(qstn_abs) != 0){
                                   qstn_abs = cosineSimilarity(params.qstn_eb, 'abstract_embedding');
                               }

                               if (Math.signum(narr_abs) != 0){
                                   narr_abs = cosineSimilarity(params.narr_eb, 'abstract_embedding');
                               }

                               return q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs + Math.log(_score);

                               """,
                    "params": {
                        "q_eb": q_eb,
                        "qstn_eb": qstn_eb,
                        "narr_eb": narr_eb,
                    }
                }
            }
        }
    }

async def generate_embedding(bc, text):
    if text:
        # Running sleep here allows us to run all embedding requests at once
        await asyncio.sleep(0.5)
        arr = np.squeeze(bc.encode([text])).tolist()
        assert len(arr) == 768

        return arr

    # Return zero vector if no text
    return [0]*768


async def run_all_queries(topics, es_client, bert_client, index_name):
    for topic_num, topic in tqdm(enumerate(topics, start=1), desc="Running Queries"):
        query = topic['query']
        question = topic['question']
        narrative = topic['narrative']

        query_embedding = await generate_embedding(bert_client, query)
        question_embedding = await generate_embedding(bert_client, question)
        narrative_embedding = await generate_embedding(bert_client, narrative)

        final_q = generate_query(query,
                                 question,
                                 narrative,
                                 query_embedding,
                                 question_embedding,
                                 narrative_embedding)

        logging.debug(f'Running query')
        logging.debug(f'{final_q}')

        # Async query so we can run multiple requests at once
        results = await es_client.search(index=index_name, body=final_q, size=1000)
        serialise_results(topic, topic_num, results)


def serialise_results(topic, topic_num, results):
    with open('results.txt', 'a+') as writer:
        for rank, result in enumerate(results['hits']['hits'], start=1):
            doc = result['_source']
            line = f"{topic_num}\tQ0\t{doc['id']}\t{rank}\t{result['_score']}\tINSERT_RUN_NAME\n"
            writer.write(line)


def parse_topics(qt_path):
    all_topics = ET.parse(qt_path).getroot()
    qtopics = [None]*30 # just incase it iterates out of order

    for topic in all_topics:
        qtopic = {
            'topic_num': '',
            'query': '',
            'question': '',
            'narrative': '',
        }

        for field in qtopic:
            try:
                qtopic[field] = topic.find(field).text
            except:
                # Skip if topic doesn't have that field
                continue

        qtopics[int(topic.attrib['number'])-1] = qtopic

    assert not (None in qtopics)

    return qtopics


@plac.annotations(
    query_topics=('path to query topics', 'positional', None, Path),
    index_name=('index name to query', 'option', None, str),
    debug=('index name to query', 'flag')
)
def main(query_topics: Path="assets/topics-rnd1.xml",
        index_name: str="covid-april-10",
        debug: bool=False):

    if debug:
        # DEBUG logs shows elasticsearch exceptions
        logging.getLogger().setLevel(logging.DEBUG)

    es_client = Elasticsearch(hosts=['localhost'])
    bc = BertClient(port=51234, port_out=51235)

    assert query_topics.exists()
    #assert asyncio.run(index_exists(es_client, index_name))

    topics = parse_topics(query_topics)

    # Run our queries asyncronously as the bottleneck is the queries themselves
    # While we wait on our queries, we can prepare the next few.
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_all_queries(topics, es_client, bc, index_name))
    loop.run_until_complete(es_client.transport.close())
    loop.close()

if __name__ == '__main__':
    plac.call(main)

