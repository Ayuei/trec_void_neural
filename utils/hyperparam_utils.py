from pathlib import Path
from sklearn.metrics import ndcg_score

def read_relevance_labels(relevance_path: Path):
    assert relevance_path.exists()
    with relevance_path.open() as f:
        return {topic_num: line.strip().split() for (topic_num, line) in enumerate(f, start=1)}


def calculate_recall_topic(doc_ids, labeled_docs):
    predicted = []

    assert len(doc_ids) > 0
    assert len(labeled_docs) > 0

    for doc_id in doc_ids:
        if doc_id.strip() in labeled_docs:
            predicted.append(1)
        else:
            predicted.append(0)

    return sum(predicted)/len(labeled_docs)


def calculate_ndcg(doc_ids, labeled_docs):
    predicted = []
    truth = []

    assert len(doc_ids) > 0
    assert len(labeled_docs) > 0

    for rank, doc_id in enumerate(doc_ids, start=1):
        rank_score = 1/rank
        if doc_id.strip() in labeled_docs:
            truth.append(1)
        else:
            truth.append(0)

        predicted.append(rank_score)

    return ndcg_score(truth, predicted)
