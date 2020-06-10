from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import argparse
from tqdm import tqdm
import torch
import os
import asyncio
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from collections import defaultdict
from typing import List
from pathlib import Path
from query import run_all_queries, parse_topics, generate_embedding
import spacy
from bert_serving.client import BertClient
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

def cosine_warm_restarts(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def no_collate_fn(data):
    (docid, qid, orig_score, query_embed, quest_embed, narr_embed, title_embedding, abstract_encoding, abstract_embedding, fulltext_encoding, fulltext_embedding, label) = zip(*data)

    abstract_encoding = pad_sequence([torch.tensor(ab) for ab in abstract_encoding], batch_first=True,
            padding_value=0)

    fulltext_encoding = pad_sequence([torch.tensor(ab) for ab in fulltext_encoding], batch_first=True,
            padding_value=0)

    return (docid, qid, torch.tensor(orig_score), torch.tensor(query_embed), torch.tensor(quest_embed), torch.tensor(narr_embed), torch.tensor(title_embedding), abstract_encoding, torch.tensor(abstract_embedding), fulltext_encoding, torch.tensor(fulltext_embedding), torch.tensor(label))

def segment_and_embed_text(nlp, bc, raw_doc, field=None):
    texts = []
    if field:
        if field not in raw_doc:
            return texts

        raw_doc = raw_doc[field]

    doc = nlp(raw_doc)

    for sent in doc.sents:
        text = sent.text.strip()

        if text:
            texts.append(text)

    if texts:
        texts = bc.encode(texts)

    return texts


class DummyCovidDatset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        (ex, label) = (self.examples[idx]['doc'], self.examples[idx]['label'])

        # doc_score, q_embed, question_embed, narr_embed, abstract_encodes, abstract_avg,
        # fulltext_encodes, fulltext_avg

        keys = ['id', 'qid', 'orig_score', 'query_embed', 'quest_embed', 'narr_embed', 'title_embedding', 'abstract_encoding',
                'abstract_embedding', 'fulltext_encoding', 'fulltext_embedding']

        to_return = []
        for key in keys:
            to_return.append(ex[key])

        to_return.append(label)

        return to_return

    def __len__(self):
        return len(self.examples)

class CovidDataset(Dataset):
    delta = 0.1  # If not judged, we still want it higher than non-relevant documents

    def __init__(self, qrels: Path, idx_name: str, result_sets: List, num_topics: int=30, num_sents=3, shuffle=True):
        assert qrels.exists()
        self.qrels = [defaultdict(lambda: 0) for _ in range(num_topics)]
        self.examples = []
        self.unlabeled = []
        self.num_sents = 3
        self.nlp = spacy.load("en_core_sci_sm", disable=['ner', 'tagger'])
        self.bc = BertClient(port=51234, port_out=51235)
        cache_file = f'cache/{idx_name}_result_sets_{num_topics}'
        if os.path.exists(cache_file):
            self.examples, self.unlabeled = pickle.load(open(cache_file, 'rb'))
        else:
            with qrels.open() as _file:
                for line in _file:
                    qid, _, docid, rel = line.split()
                    self.qrels[int(qid)-1][docid] = int(rel) # Binary

            for i, result_set in enumerate(result_sets):
                if os.path.exists(f'{cache_file}_{i}'):
                    print(f"Loaded {cache_file}_{i}")
                    self.examples, self.unlabeled = pickle.load(open(f'{cache_file}_{i}', 'rb'))
                else:
                    self.add_result_set(result_set)
                    pickle.dump([self.examples, self.unlabeled],
                            open(f'{cache_file}_{i}', 'wb+'))

        if shuffle:
            import random

            r = random.Random(42)
            r.shuffle(self.unlabeled)

            self.examples = self.examples + self.unlabeled[:int(len(self.examples) * 0.33)]
            self.train, self.validation = torch.utils.data.random_split(self.examples,
                    [int(0.7*len(self.examples)),
                     len(self.examples)-int(0.7*len(self.examples))])

            self.train = DummyCovidDatset(self.train)
            self.validation = DummyCovidDatset(self.validation)
        else:
            self.examples = self.examples + self.unlabeled
            self.examples = DummyCovidDatset(self.examples)

    def add_result_set(self, result_set):
        qid, q_embed, quest_embed, narr_embed, results = result_set
        assert int(qid) > 0
        qid_qrels = self.qrels[int(qid)-1]

        # for rank, result in enumerate(results['hits']['hits'], start=1):
        # doc = result['_source']

        for rank, result in tqdm(enumerate(results['hits']['hits'], start=1), desc=f'Transforming Result Set {qid}'):
            doc = result['_source']
            score = result['_score']
            docid = doc['id']

            # 'id', 'title', 'fulltext', 'abstract', 'date', 'title_embedding', 'abstract_embedding'

            doc['abstract_encoding'] = segment_and_embed_text(self.nlp, self.bc, doc, field='abstract')
            doc['fulltext_encoding'] = segment_and_embed_text(self.nlp, self.bc, doc, field='fulltext')

            if not len(doc['abstract_encoding']):
                doc['abstract_encoding'] = [[0]*768]*self.num_sents

            if not len(doc['fulltext_encoding']):
                doc['fulltext_encoding'] = [[0]*768]*self.num_sents

            doc['fulltext_embedding'] = np.mean(doc['fulltext_encoding'], axis=0)
            doc['orig_score'] = score

            doc['query_embed'] = q_embed
            doc['quest_embed'] = quest_embed
            doc['narr_embed'] = narr_embed
            doc['qid'] = qid

            if docid in qid_qrels.keys():
                label = qid_qrels[docid]
                self.examples.append({
                    'doc': doc,
                    'label': float(label)/2})
            else:
                self.unlabeled.append({
                    'doc': doc,
                    'label': self.delta})

        assert len(self.unlabeled) > 0

    def __getitem__(self, idx):
        (ex, label) = (self.examples[idx]['doc'], self.examples[idx]['label'])

        # doc_score, q_embed, question_embed, narr_embed, abstract_encodes, abstract_avg,
        # fulltext_encodes, fulltext_avg

        keys = ['id', 'qid', 'orig_score', 'query_embed', 'quest_embed', 'narr_embed', 'title_embedding', 'abstract_encoding',
                'abstract_embedding', 'fulltext_encoding', 'fulltext_embedding']

        to_return = []
        for key in keys:
            to_return.append(ex[key])

        to_return.append(label)

        return to_return

    def __len__(self):
        return len(self.examples)


class CovidRel(LightningModule):
    def __init__(self, hparams, dataset):
        super().__init__()
        # do this to save all arguments in any logger (tensorboard)
        self.hparams = hparams
        self.hidden_dim = 768
        self.num_sents = 3 
        # score(q, t), score_k(q, a in A), score(q, a), score(q, abs), score(q, )
        self.layer_1 = torch.nn.Linear(28, 1)
        self.layer_2 = torch.nn.Sigmoid()
        self.dataset = dataset
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.loss = torch.nn.MSELoss()

    def forward(self, doc_score, query_embed, quest_embed, narr_embed, title_embedding,
            abstract_encoding, abstract_embedding, fulltext_encoding, fulltext_embedding,
            rank_scores=False):

        scores = [18*torch.unsqueeze(doc_score, dim=1)]

        for q_e in [query_embed, quest_embed, narr_embed]:
            tmp = {
                "rel_qry_ttl" : self.cos(q_e, title_embedding),
                "rel_qry_absa" : self.cos(q_e, abstract_embedding),
                "rel_qry_fta": self.cos(q_e, fulltext_embedding),
            }

            #(Pdb) self.cos(q_e[0], abstract_encoding[0][0])
            #tensor(0.3256)
            #(Pdb) self.cos(q_e[0], abstract_encoding[0][1])
            #tensor(0.3200)
            #(Pdb) self.cos(q_e[0], abstract_encoding[0][2])
            #tensor(0.5321)
            #(Pdb) self.cos(q_e[0], abstract_encoding[0][3])


            abstract_encodes_scores = []

            for i, (q, encodes) in enumerate(zip(q_e, abstract_encoding)):
                temp = []
                for sent in encodes:
                    temp.append(self.cos(q, sent))

                for i in range(3):
                    temp.append(torch.tensor(0.000))

                values, _ = torch.topk(torch.stack(temp), 3)

                abstract_encodes_scores.append(values)

            tmp["rel_qry_abs"] = torch.stack(abstract_encodes_scores)

            fulltext_encodes_scores = []

            for i, (q, encodes) in enumerate(zip(q_e, fulltext_encoding)):
                temp = []
                for sent in encodes:
                    temp.append(self.cos(q, sent))

                for i in range(3):
                    temp.append(torch.tensor(0.000))

                values, _ = torch.topk(torch.stack(temp), 3)

                fulltext_encodes_scores.append(values)

            tmp["rel_qry_ft"] = torch.stack(fulltext_encodes_scores)

            if not rank_scores:
                temp = torch.cat([torch.unsqueeze(tmp["rel_qry_ttl"], dim=1),
                              torch.unsqueeze(tmp["rel_qry_absa"], dim=1),
                              torch.unsqueeze(tmp["rel_qry_fta"], dim=1),
                              tmp["rel_qry_abs"],
                              tmp["rel_qry_ft"]], dim=1)

            else:
                temp = torch.cat([tmp["rel_qry_abs"],
                                  tmp["rel_qry_ft"]], dim=1)

            scores.append(temp)

        scores = torch.cat(scores, dim=1)

        if rank_scores:
            return torch.sum(scores, dim=1)

        return F.sigmoid(self.layer_1(scores))

    def train_dataloader(self):
        return DataLoader(self.dataset.train, batch_size=self.hparams.batch_size, collate_fn=no_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset.validation, batch_size=self.hparams.batch_size, collate_fn=no_collate_fn)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        #num_training_steps = len(self.dataset) * 20
        #num_warmup_steps = int(0.1 * num_training_steps)
        #sched = cosine_warm_restarts(opt, num_warmup_steps, num_training_steps)
        #return [opt], [sched]

        return opt

    def training_step(self, batch, batch_idx):
        # implement your own
        (_, _, doc_score, query_embed, quest_embed, narr_embed, title_embedding, abstract_encoding,
        abstract_embedding, fulltext_encoding, fulltext_embedding, labels) = batch

        out = self(doc_score=doc_score,
                   query_embed=query_embed,
                   quest_embed=quest_embed,
                   narr_embed=narr_embed,
                   title_embedding=title_embedding,
                   abstract_encoding=abstract_encoding,
                   abstract_embedding=abstract_embedding,
                   fulltext_encoding=fulltext_encoding,
                   fulltext_embedding=fulltext_embedding)

        loss = self.loss(torch.squeeze(out), labels)

        logger_logs = {'training_loss': loss} # optional (MUST ALL BE TENSORS)

        # if using TestTubeLogger or TensorBoardLogger you can nest scalars
        #logger_logs = {'losses': logger_logs} # optional (MUST ALL BE TENSORS)

        output = {
            'loss': loss, # required
            'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
            'log': logger_logs
        }

        # return a dict
        return output

    def validation_step(self, batch, batch_idx):
        # implement your own
        (_, _, doc_score, query_embed, quest_embed, narr_embed, title_embedding, abstract_encoding,
        abstract_embedding, fulltext_encoding, fulltext_embedding, labels) = batch

        out = self(doc_score=doc_score,
                   query_embed=query_embed,
                   quest_embed=quest_embed,
                   narr_embed=narr_embed,
                   title_embedding=title_embedding,
                   abstract_encoding=abstract_encoding,
                   abstract_embedding=abstract_embedding,
                   fulltext_encoding=fulltext_encoding,
                   fulltext_embedding=fulltext_embedding)

        loss = self.loss(torch.squeeze(out), labels)

        output = {
            'val_loss': loss
        }

        # return a dict
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()

        return {'avg_val_loss': val_loss_mean,
                'step': self.current_epoch}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_sents', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--prepare_data', action='store_true')
        return parser


def train_model():
    # default used by the Trainer
    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        patience=2,
        strict=False,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd()+"/model_outputs/",
        save_top_k=True,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
    )


    topics = parse_topics("./assets/topics-rnd1.xml")
    loop = asyncio.get_event_loop()
    index = "covid-april-10"
    #results = loop.run_until_complete(
    #    run_all_queries(
    #            topics,
    #            index_name=index,
    #            cosine_weights=[1]*6,
    #            query_weights=[1]*12,
    #            return_queries=True,
    #            size=1500,
    #    )
    #)

    #loop.close()
    results = [None]*30

    parser = argparse.ArgumentParser()
    parser = CovidRel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dataset = CovidDataset(Path("./assets/qrels-rnd1.txt"), index, results, num_topics=len(results))
    model = CovidRel(hparams, dataset)

    if hparams.prepare_data:
        return

    trainer = Trainer(max_epochs=100, gpus=1, auto_scale_batch='binsearch', early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

def rerank_scores(collection, index):
    all_scores = []
    all_ids = []
    all_qids = []

    parser = argparse.ArgumentParser()
    parser = CovidRel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dataset = CovidDataset(Path("./assets/empty_qrels.txt"), index, collection,
            num_topics=len(collection), shuffle=False)

    model = CovidRel(hparams, dataset)
    print(len(dataset.examples))
    dataloader = DataLoader(dataset.examples, batch_size=1, collate_fn=no_collate_fn,
            num_workers=1)

    for batch in tqdm(dataloader, total=len(dataset.examples) // 1, desc="Getting scores"):
        (docid, qid, doc_score, query_embed, quest_embed, narr_embed, title_embedding, abstract_encoding,
        abstract_embedding, fulltext_encoding, fulltext_embedding, _) = batch

        preds = model(doc_score, query_embed, quest_embed, narr_embed, title_embedding,
                abstract_encoding, abstract_embedding, fulltext_encoding, fulltext_embedding,
                rank_scores=True)

        all_scores.extend(preds)
        all_ids.extend(docid)
        all_qids.extend(qid)

    import pdb; pdb.set_trace()
    all_scores, all_qids, all_ids = torch.save([all_scores, all_qids, all_ids], "predict_save_new.pt")
    all_scores, all_qids, all_ids = torch.load("predict_save_new.pt", map_location=torch.device('cpu'))

    combined_dict = defaultdict(lambda: [])

    for score, _id, qid in zip(all_scores, all_ids, all_qids):
        combined_dict[qid].append([_id, score.item()])

    for qid in combined_dict:
        combined_dict[qid].sort(key = lambda k: k[1], reverse=True) # sort by scores
        combined_dict[qid] = combined_dict[qid][:1000]

    for topic_num in tqdm(combined_dict.keys(), desc='Serializing'):
        serialize_rerank(combined_dict[topic_num], int(topic_num))

def serialize_rerank(docs, topic_num):
    with open('new_logicistic_rerank_results.txt', 'a+') as writer:
        for rank, (docid, score) in enumerate(docs, start=1):
            line = f"{topic_num}\tQ0\t{docid}\t{rank}\t{score}\tINSERT_RUN_NAME\n"
            writer.write(line)


if __name__ == '__main__':
    #train_model()
    #topics = parse_topics("./assets/topics-rnd3.xml")
    #loop = asyncio.get_event_loop()
    index = "covid-may-19-fulltext_embed"

    #results = loop.run_until_complete(
    #    run_all_queries(
    #            topics,
    #            index_name=index,
    #            cosine_weights=[1]*6,
    #            query_weights=[1]*12,
    #            return_queries=True,
    #            size=1500,
    #    )
    #)

    #import dill; dill.dump(results, open("temp.temp", "wb+"))
    import dill
    results = dill.load(open("temp.temp", "rb"))

    #loop.close()
    rerank_scores(results, index)
