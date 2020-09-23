import math
import logging

class WeightTranslator:
    def __init__(self, weights, translator):
        self.weights = weights
        self.translator = translator

    def get_weight(self, qfield=None, facet=None, return_index=False):
        assert qfield in self.translator or qfield is None, qfield
        assert qfield != facet # Both cannot be None
        indexes = []

        if facet is None:
            indexes = [self.translator[qfield][f] for f in self.translator[qfield]]

        if qfield is None:
            indexes = [self.translator[qf][facet] for qf in self.translator]

        if indexes:
            if return_index:
                return indexes
            return [self.weights[i] for i in indexes]
        else:
            index = self.translator[qfield][facet]
            if return_index:
                index
            return self.weights[index]

    def get_all_weights(self):
        return self.weights

    def exclude(self, qfield=None, facet=None):
        #print(qfield, facet)
        #print(self.weights)
        #print(self.translator)
        indexes = self.get_weight(qfield, facet, return_index=True)

        for index in indexes:
            self.weights[index] = 0

    def set_weight(self, qfield=None, facet=None, weight=1.0):
        # print(qfield, facet)
        indexes = self.get_weight(qfield, facet, return_index=True)

        for index in indexes:
            self.weights[index] = weight 


class QueryWeightTranslator(WeightTranslator):
    fields = ["query", "question", "narrative", "expansion"]
    facets = ["title", "abstract", "fulltext"]

    def __init__(self, weights=None):
        d = {}
        i = 0
        for qfield in self.fields:
            d[qfield] = {}
            for facet in self.facets:
                d[qfield][facet] = i
                i += 1

        if not weights:
            weights = [1 for i in range(len(self.fields) * len(self.facets))]

        super().__init__(weights, d)


class CosineWeightTranslator(WeightTranslator):
    fields = ["query", "question", "narrative"]
    facets = ["title", "abstract", "fulltext"]

    def __init__(self, weights=None):
        d = {}
        i = 0
        for qfield in ["query", "question", "narrative"]:
            d[qfield] = {}
            for facet in ["title", "abstract", "fulltext"]:
                d[qfield][facet] = i
                i += 1

        if not weights:
            weights = [1 for i in range(len(self.fields) * len(self.facets))]

        super().__init__(weights, d)


class QNorm:
    def __init__(self, gold_standard, qwt, cwt):
        last_line = open(gold_standard, 'r').readlines()[-1]
        num_topics = int(last_line.split()[0])
        num_docs_per_topic = int(last_line.split()[3])

        assert num_docs_per_topic == 1000

        scores = [0 for _ in range(num_topics)]

        with open(gold_standard, 'r') as f:
            for i, line in enumerate(f):
                if i % num_docs_per_topic == 0:
                    indx = int(line.split()[0]) - 1 # Topic number - 1 = index
                    scores[indx] = float(line.split()[4])

        self.scores = scores
        self.qwt = qwt
        self.cwt = cwt

    def get_norm_weight_by_query(self, qid, estimate_ceiling=False):
        return self.get_norm_weight(self.qwt, self.cwt, bm25_ceiling=self.scores[int(qid)-1], 
        estimate_ceiling=estimate_ceiling)

    @classmethod
    def get_norm_weight(cls, qwt, cwt, bm25_ceiling=100, estimate_ceiling=False):
        qw_len = len(qwt.get_all_weights())
        qw_non_zero = len(list(filter(lambda k: k>0, qwt.get_all_weights())))

        if estimate_ceiling:
            bm25_ceiling = qw_non_zero/qw_len * bm25_ceiling 

        cosine_ceiling = len(list(filter(lambda k: k>0, cwt.get_all_weights())))

        # Analytical solution for getting log base: 
        # n_score - log(bm25_score)/log(x) = 0
        # Solve for x
        return bm25_ceiling**(1/float(cosine_ceiling))
