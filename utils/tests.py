import unittest
from train_utils import (
    WeightTranslator,
    QueryWeightTranslator,
    CosineWeightTranslator,
    QNorm,
)


class TestWeightTranslator(unittest.TestCase):
    def setUp(self):
        self.qwt = QueryWeightTranslator()
        self.cwt = CosineWeightTranslator()

        self.wts = [self.qwt, self.cwt]

    def test_get_all_weights(self):
        for wt in self.wts:
            dummy_length = len(wt.fields) * len(wt.facets)
            dummy_weights = [1 for i in range(dummy_length)]

            self.assertEqual(wt.get_all_weights(), dummy_weights)

    def test_get_weight_simple(self):
        for wt in self.wts:
            dummy_length = len(wt.fields) * len(wt.facets)
            for facet in wt.facets:
                for field in wt.fields:
                    self.assertEqual(wt.get_weight(field, facet), 1)

    def test_get_weight_complex(self):
        field_translator = {"query": 2, "question": 4, "narrative": 3}

        cwt_weights = [2, 2, 2, 4, 4, 4, 3, 3, 3]
        qwt_weights = [2, 2, 2, 4, 4, 4, 3, 3, 3, 1, 1, 1]

        for fld in field_translator.keys():
            field_v = field_translator[fld]

            for wt in self.wts:
                wt.set_weight(qfield=fld, facet=None, weight=field_v)

        self.assertEqual(self.cwt.get_all_weights(), cwt_weights)
        self.assertEqual(self.qwt.get_all_weights(), qwt_weights)

        facet_translator = {
            "title": 8,
            "abstract": 5,
        }

        cwt_weights = [8, 5, 2, 8, 5, 4, 8, 5, 3]
        qwt_weights = [8, 5, 2, 8, 5, 4, 8, 5, 3, 8, 5, 1]

        for fct in facet_translator.keys():
            v = facet_translator[fct]

            for wt in self.wts:
                wt.set_weight(qfield=None, facet=fct, weight=v)

        self.assertEqual(self.cwt.get_all_weights(), cwt_weights)
        self.assertEqual(self.qwt.get_all_weights(), qwt_weights)


class Test(unittest.TestCase):
    def setUp(self):
        self.qwt = QueryWeightTranslator()
        self.cwt = CosineWeightTranslator()
        self.qnorm = QNorm("testfile", self.qwt, self.cwt)

    def test_init(self):
        scores = [
            102.710175,
            159.76755,
            125.15036,
            74.87103,
            187.1404,
        ]

        last_score = 182.38461

        self.assertEqual(len(self.qnorm.scores), 50)

        for score, qscore in zip(scores, self.qnorm.scores):
            self.assertEqual(score, qscore)

        self.assertEqual(self.qnorm.scores[-1], last_score)

    def test_norm_bounds(self):
        import math

        qnorm = self.qnorm
        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights())
            score = math.log(score, norm) + cosine_ceiling

            self.assertAlmostEqual(score, cosine_ceiling * 2, delta=1e-10)
            self.assertGreaterEqual(score, 0)

    def test_per_query(self):
        import math

        qnorm = self.qnorm

        for i in range(1, 50):
            norm = qnorm.get_norm_weight_by_query(i)
            cosine_ceiling = len(self.cwt.get_all_weights())
            score = math.log(qnorm.scores[i - 1], norm) + cosine_ceiling

            self.assertAlmostEqual(score, cosine_ceiling * 2, delta=1e-10)
            self.assertGreaterEqual(score, 0)

    def test_bounds_field(self):
        import math

        qnorm = self.qnorm
        self.qwt.set_weight(qfield="query", weight=0)
        self.cwt.set_weight(qfield="query", weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights()) - 3
            score = math.log(score) / math.log(norm)
            delta = score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(score, 0)

    def test_bounds_facet(self):
        import math

        qnorm = self.qnorm
        self.qwt.set_weight(facet="fulltext", weight=0)
        self.cwt.set_weight(facet="fulltext", weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights()) - 3
            new_score = math.log(score) / math.log(norm)
            delta = new_score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(new_score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(new_score, 0)

    def test_bounds_facet_cosine(self):
        import math

        qnorm = self.qnorm
        self.qwt.set_weight(facet="fulltext", weight=0)
        # self.cwt.set_weight(facet='fulltext', weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights())
            new_score = math.log(score) / math.log(norm)
            delta = new_score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(new_score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(new_score, 0)

    def test_bounds_field_cosine(self):
        import math

        qnorm = self.qnorm
        self.qwt.set_weight(qfield="query", weight=0)
        # self.cwt.set_weight(facet='fulltext', weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights())
            new_score = math.log(score) / math.log(norm)
            delta = new_score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(new_score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(new_score, 0)

    def test_bounds_facet_cosine_opp(self):
        import math

        qnorm = self.qnorm
        # self.qwt.set_weight(field='query', weight=0)
        self.cwt.set_weight(facet="fulltext", weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights()) - 3
            new_score = math.log(score) / math.log(norm)
            delta = new_score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(new_score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(new_score, 0)

    def test_bounds_field_cosine_opp(self):
        import math

        qnorm = self.qnorm
        # self.qwt.set_weight(field='query', weight=0)
        self.cwt.set_weight(field="query", weight=0)

        for score in qnorm.scores:
            norm = qnorm.get_norm_weight(self.qwt, self.cwt, score)
            cosine_ceiling = len(self.cwt.get_all_weights()) - 3
            new_score = math.log(score) / math.log(norm)
            delta = new_score - (math.log(score) / math.log(norm + 1e-3))

            self.assertAlmostEqual(new_score, cosine_ceiling, delta=delta)
            self.assertGreaterEqual(new_score, 0)


if __name__ == "__main__":
    unittest.main()
