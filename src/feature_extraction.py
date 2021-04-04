import spacy
from nltk import ngrams

class SpacyFeaturizer:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")

    def featurize(self, segments):
        feature_dicts = []
        docs = self.nlp.pipe(segments)
        for doc in docs:
            feature_dicts.append(self.get_features(doc))
        return feature_dicts

    def get_features(self, doc, use_ngrams=True):
        features = {}

        # for ngrams
        pos_tags = []
        dep_tags = []

        for token in doc:
            # Morphology: tense, person, mood, etc
            for key,value in token.morph.to_dict().items():
                features[key+value] = 1
            # Lemmas
            features[token.lemma_] = 1
            # POS
            features[token.pos_] = 1
            # Dep
            features[token.dep_] = 1

            #features.update(token.vector)
            pos_tags.append(token.pos_)
            #dep_tags.append(token.dep_)

        if use_ngrams:
            # Generate ngrams of length 3
            sentence_ngrams = ngrams(pos_tags,3)
            for ngram in sentence_ngrams:
                features[" ".join(ngram)] = 1


            sentence_ngrams = ngrams(dep_tags,3)
            for ngram in sentence_ngrams:
                features[" ".join(ngram)] = 1

        return features
