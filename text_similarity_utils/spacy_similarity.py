import spacy
import numpy as np

class SpacySimilarity:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def _similarity(self, s1, s2):
        s1 = self.nlp(s1)
        s2 = self.nlp(s2)
        s1 = self.nlp(' '.join([str(t) for t in s1 if t.pos_ in ['NOUN', 'PROPN', 'VERB']]))
        s2 = self.nlp(' '.join([str(t) for t in s2 if t.pos_ in ['NOUN', 'PROPN', 'VERB']]))

        return s1.similarity(s2)

    def compute_score(self, dset, query):
        """
        dset: List[List[str]]
        query: str
        """
        scores = np.zeros(len(dset))
        for i, s in enumerate(dset):
            # s is a List[str], that is all the strings associated to a certain motion.
            # take the maximum (the annotator that more agrees with the given query)
            scores[i] = max([self._similarity(query, t) for t in s])

        return scores