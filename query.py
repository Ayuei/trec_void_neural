import plac
import os
from aioelasticsearch import Elasticsearch
import asyncio
from xml.etree import ElementTree as ET
from pathlib import Path
from bert_serving.client import BertClient
import logging
from tqdm import tqdm
import numpy as np
from joblib import Memory
from utils import hyperparam_utils
from hyperopt import hp
from ray.tune import track
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from utils.train_utils import QueryWeightTranslator, CosineWeightTranslator, QNorm


# Final score = log(BM25 of query/question/narrative) + cosine(query, doc_title) + cosine(question, doc_title) + cosine(narr, doc_title)
# + cosine(query, doc_abstract) + cosine(question, doc_abstract) + cosine(narr, doc_abstract)
# Log on bm25 is used as it generally gives a score from 0 - 6 which is the range of our cosine similarity
def generate_query(q, qstn, narr, q_eb, qstn_eb, narr_eb, cosine_weights=[1]*9,
        query_weights=[1]*12,
expansion="disease severe acute respiratory syndrome coronavirus treatment virus covid-19 sars-cov-2 covid sars", norm_weight=2.15):
    assert len(query_weights) == 12
    assert len(cosine_weights) == 9


    expansion = '' # set expansion to nothing for submission

    return {
        "_source": {
            "excludes": ["*.abstract_embedding_array", "*.fulltext_embedding_array"]
        },
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        # Match on title, abstract and fulltext on all three fields
                        # Weights should be added later
                        "should": [
                            {"match": {"title": {"query": q, "boost": query_weights[0]}}},
                            {"match": {"title": {"query": qstn, "boost": query_weights[1]}}},
                            {"match": {"title": {"query": narr, "boost": query_weights[2]}}},
                            {"match": {"title": {"query": expansion, "boost": query_weights[3]}}},
                            {"match": {"abstract": {"query": q, "boost": query_weights[4]}}},
                            {"match": {"abstract": {"query": qstn, "boost": query_weights[5]}}},
                            {"match": {"abstract": {"query": narr, "boost": query_weights[6]}}},
                            {"match": {"abstract": {"query": expansion, "boost": query_weights[7]}}},
                            {"match": {"fulltext": {"query": q, "boost": query_weights[8]}}},
                            {"match": {"fulltext": {"query": qstn, "boost": query_weights[9]}}},
                            {"match": {"fulltext": {"query": narr, "boost": query_weights[10]}}},
                            {"match": {"fulltext": {"query": expansion, "boost": query_weights[11]}}},
                        ],
                        "filter": {
                            "range": {
                                "date": {"gte": "2019-12-31"},
                            }
                        }
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
                               def weights = params.weights;
                               // If document score is zero, don't do score calculation
                               // Filter query has set it to zero
                               if (Math.signum(_score) == 0){
                                   return 0.0;
                               }

                               if (params.norm_weight < 0.0) {
                                   return _score;
                               }

                               double q_t = dotProduct(params.q_eb, 'title_embedding');
                               double qstn_t = dotProduct(params.qstn_eb, 'title_embedding');
                               double narr_t = dotProduct(params.narr_eb, 'title_embedding');

                               double q_abs = dotProduct(params.q_eb, 'abstract_embedding');
                               double qstn_abs = dotProduct(params.qstn_eb, 'abstract_embedding');
                               double narr_abs = dotProduct(params.narr_eb, 'abstract_embedding');

                               //double q_tb = 0.0;
                               //double qstn_tb = 0.0;
                               //double narr_tb = 0.0;

                               //try{
                               //     q_tb = dotProduct(params.q_eb, 'fulltext_embedding');
                               //     qstn_tb = dotProduct(params.qstn_eb, 'fulltext_embedding');
                               //     narr_tb = dotProduct(params.narr_eb, 'fulltext_embedding');
                               // } catch(Exception e){
                               // }

                               if (Math.signum(q_t) != 0){
                                   q_t = weights[0]*cosineSimilarity(params.q_eb, 'title_embedding') + params.offset;
                               }

                               if (Math.signum(qstn_t) != 0){
                                   qstn_t = weights[1]*cosineSimilarity(params.qstn_eb, 'title_embedding') + params.offset;
                               }

                               if (Math.signum(narr_t) != 0){
                                   narr_t = weights[2]*cosineSimilarity(params.narr_eb, 'title_embedding')+params.offset;
                               }

                               if (Math.signum(q_abs) != 0){
                                   q_abs = weights[3]*cosineSimilarity(params.q_eb, 'abstract_embedding')+params.offset;
                               }

                               if (Math.signum(qstn_abs) != 0){
                                   qstn_abs = weights[4]*cosineSimilarity(params.qstn_eb, 'abstract_embedding')+params.offset;
                               }

                               if (Math.signum(narr_abs) != 0){
                                   narr_abs = weights[5]*cosineSimilarity(params.narr_eb, 'abstract_embedding')+params.offset;
                               }

                               //if (Math.signum(q_tb) != 0){
                               //    q_tb = weights[6]*cosineSimilarity(params.q_eb, 'fulltext_embedding')+1.0;
                               //}

                               //if (Math.signum(qstn_tb) != 0){
                               //    qstn_tb = weights[7]*cosineSimilarity(params.qstn_eb, 'fulltext_embedding')+1.0;
                               //}

                               //if (Math.signum(narr_tb) != 0){
                               //    narr_tb = weights[8]*cosineSimilarity(params.narr_eb, 'fulltext_embedding')+1.0;
                               //}

                               // return q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs + Math.log(_score)/Math.log(1.66); // 2.15
                               // return (q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs)/params.divisor + Math.log(_score+1)/Math.log(params.norm_weight); // 2.15 // 1.66
                               // return Math.log(_score+1)/Math.log(params.norm_weight);
                               // return (q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs)/params.divisor;

                               // return _score;
                               return q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs - params.reduce_offset + Math.log(_score)/Math.log(params.norm_weight); // 2.15
                               """,
                    "params": {
                        "q_eb": q_eb,
                        "qstn_eb": qstn_eb,
                        "narr_eb": narr_eb,
                        "weights": cosine_weights,
                        "reduce_offset": len(cosine_weights)-sum(cosine_weights),
                        "norm_weight": norm_weight,
                        "divisor": 1.0,
                        "offset": 1.0,
                    }
                }
            }
        }
    }

# @use_memory_cache
def generate_embedding(bc, text):
    if text:
        # Running sleep here allows us to run all embedding requests at once
        arr = np.squeeze(bc.encode([text])).tolist()
        assert len(arr) == 768

        return arr

    # Return zero vector if no text
    return [0]*768


async def run_all_queries(topics, index_name,
        cosine_weights, query_weights,
        size=2383, tune_model:bool=False,
        return_queries=False, output_file="results.txt", bert_inport=51234, norm_weight=2.15,
        qnorm=None):

    es_client = Elasticsearch(timeout=600)
    bert_client = BertClient(port=bert_inport, port_out=bert_inport+1)

    if tune_model:
        path = Path('/home/ngu143/Projects/trec_void_neural/assets/rl_labels.txt')
        assert path.exists()
        labels_func = hyperparam_utils.read_relevance_labels
        labels = labels_func(path)

    moving_average = []
    ret_results = []

    for topic_num, topic in tqdm(enumerate(topics, start=1), desc="Running Queries",
            disable=tune_model):

        if topic is None:
            continue

        query = topic['query']
        question = topic['question']
        narrative = topic['narrative']

        query_embedding = generate_embedding(bert_client, query)
        question_embedding = generate_embedding(bert_client, question)
        narrative_embedding = generate_embedding(bert_client, narrative)

        if qnorm:
            norm_weight = qnorm.get_norm_weight_by_query(topic_num, estimate_ceiling=False)

        final_q = generate_query(query,
                                 question,
                                 narrative,
                                 query_embedding,
                                 question_embedding,
                                 narrative_embedding,
                                 cosine_weights,
                                 query_weights,
                                 norm_weight=norm_weight)

        logging.debug(f'Running query')
        logging.debug(f'{final_q}')

        # Async query so we can run multiple requests at once
        results = await es_client.search(index=index_name, body=final_q, size=size)
        if not tune_model:
            if not return_queries:
                serialise_results(topic, topic_num, results, output_file)
            else:
                ret_results.append([topic_num,
                                    query_embedding,
                                    question_embedding,
                                    narrative_embedding,
                                    results])
        else:
            doc_ids = [doc['_source']['id'] for doc in results['hits']['hits']]
            score = hyperparam_utils.calculate_recall_topic(doc_ids, labels[topic_num])
            moving_average.append(score)
            #tune.track.log(moving_average_recall=sum(moving_average)/len(moving_average))

    #print(sum(moving_average)/len(moving_average))

    bert_client.close()
    await es_client.close()

    return ret_results


def serialise_results(topic, topic_num, results, output_file):
    with open(output_file, 'a+') as writer:
        for rank, result in enumerate(results['hits']['hits'], start=1):
            doc = result['_source']
            line = f"{topic_num}\tQ0\t{doc['id']}\t{rank}\t{result['_score']}\tINSERT_RUN_NAME\n"
            writer.write(line)


# @use_memory_cache
def parse_topics(qt_path):
    if qt_path.name.endswith(".pkl"):
        import pickle
        print("Loading queries from pickle")
        return pickle.load(qt_path.open('rb'))

    all_topics = ET.parse(qt_path).getroot()
    qtopics = [None]*len(all_topics.findall('topic')) # just incase it iterates out of order

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

        # print(topic.attrib['number'])
        qtopics[int(topic.attrib['number'])-1] = qtopic

    assert not (None in qtopics)

    return qtopics


def run_query_wrapper(kwargs):
    query_keys = [f'w{i}' for i in range(1, 13)]
    cosine_keys = [f'w{i}' for i in range(13, 19)]
    topics = kwargs.pop('topics')
    index_name = kwargs.pop('index_name')
    cosine_weights = [kwargs[key] for key in cosine_keys]
    query_weights = [kwargs[key] for key in query_keys]

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_all_queries(topics, index_name,
        cosine_weights, query_weights, size=50, tune_model=True))
    loop.close()


def tune_wrapper(topics, index_name):
    space = {
            "w1": hp.uniform('w1', 0.1, 2),
            "w2": hp.uniform('w2', 0.1, 2),
            "w3": hp.uniform('w3', 0.1, 2),
            "w4": hp.uniform('w4', 0.1, 2),
            "w5": hp.uniform('w5', 0.1, 2),
            "w6": hp.uniform('w6', 0.1, 2),
            "w7": hp.uniform('w7', 0.1, 2),
            "w8": hp.uniform('w8', 0.1, 2),
            "w9": hp.uniform('w9', 0.1, 2),
            "w10": hp.uniform('w10', 0.1, 2),
            "w11": hp.uniform('w11', 0.1, 2),
            "w12": hp.uniform('w12', 0.1, 2),
            "w13": hp.uniform('w13', 0.1, 2),
            "w14": hp.uniform('w14', 0.1, 2),
            "w15": hp.uniform('w15', 0.1, 2),
            "w16": hp.uniform('w16', 0.1, 2),
            "w17": hp.uniform('w17', 0.1, 2),
            "w18": hp.uniform('w18', 0.1, 2),
    }

    hyperopt_search = HyperOptSearch(
            space,
            max_concurrent=2,
            reward_attr="moving_average_recall",
            n_initial_points=60,
            random_state_seed=42)

    analysis = tune.run(
        run_query_wrapper,
        name="tune_covid_MA_RECALL_2",
        num_samples=2000,
        resources_per_trial={'cpu':1},
        scheduler=AsyncHyperBandScheduler(metric="moving_average_recall", mode="max", grace_period=3),
        search_alg=hyperopt_search,
        config={'topics': topics, 'index_name': index_name},
        verbose=1
    )


@plac.annotations(
    query_topics=('path to query topics', 'option', None, Path),
    index_name=('index name to query', 'option', None, str),
    debug=('activate debug logger', 'flag'),
    tune=('perform hyperparameter tuning', 'flag'),
    output_file=('output_file', 'option', None, str),
    bert_inport=('BC port in', 'option', None, int),
    norm_weight=('BM25 normalization weight', 'option', None, float),
    bm25_only=('BM25 results only', 'flag'),
    exclude=('Exclude facets or query fields', 'option')
)
def main(query_topics: Path="assets/topics-rnd1.xml",
        index_name: str="covid-april-10-dated",
        debug: bool=False,
        tune=False,
        bert_inport=51234,
        output_file="results.txt",
        norm_weight=2.15,
        bm25_only: bool=False,
        exclude=None):

    print(norm_weight)
    if debug:
        # DEBUG logs shows elasticsearch exceptions
        logging.getLogger().setLevel(logging.DEBUG)

    #es_client = await Elasticsearch(hosts=['localhost'])

    assert query_topics.exists()
    #assert asyncio.run(index_exists(es_client, index_name))

    topics = parse_topics(query_topics)

    if tune:
        tune_wrapper(topics, index_name)
    else:
        # Run our queries asyncronously as the bottleneck is the queries themselves
        # While we wait on our queries, we can prepare the next few.

        qwt = QueryWeightTranslator()
        cwt = CosineWeightTranslator()

        if exclude:
            print("Excluding", exclude)
            qwt.exclude(exclude)
            # qwt.exclude(facet="fulltext")
            # cwt.exclude(exclude)

        cwt.exclude(facet='fulltext')
        cwt.exclude(facet='title')
        loop = asyncio.get_event_loop()

        qnorm = None
        if norm_weight < 0 or bm25_only:
            print("Override: Using automatic normalizer for BM-25 weight")
            print("Running trial run first to get weights")
            if os.path.exists("weights_file.temp"):
                import shutil
                open("weights_file.temp", 'w+').close() #Empty file

            loop.run_until_complete(run_all_queries(topics, index_name,
                cosine_weights=cwt.get_all_weights(),
                query_weights=qwt.get_all_weights(),
                output_file="weights_file.temp",
                bert_inport=bert_inport,
                norm_weight=norm_weight,
                qnorm=None))

            print("Finished collecting statistics, running")
            qnorm = QNorm("weights_file.temp", qwt=qwt, cwt=cwt)

        if bm25_only:
            print("Yielding only bm25 results")
            import shutil
            shutil.move("weights_file.temp", output_file)
            loop.close()
            import sys; sys.exit(0)

        loop.run_until_complete(run_all_queries(topics, index_name,
            cosine_weights=cwt.get_all_weights(),
            query_weights=qwt.get_all_weights(),
            output_file=output_file,
            bert_inport=bert_inport,
            norm_weight=norm_weight,
            qnorm=qnorm))
        loop.close()

if __name__ == '__main__':
    plac.call(main)

