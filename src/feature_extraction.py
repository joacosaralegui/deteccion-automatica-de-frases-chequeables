import spacy

from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np

# Today reference date
today = datetime.today()

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

class SpacyFeaturizer:
    def __init__(self, version="new"):
        # Spacy
        if version == "chq":
            self.nlp = spacy.load("es_core_news_lg")
        elif version == "ff":
            self.nlp_use = spacy.load("xx_use_lg")
        else:
            self.nlp_use = spacy.load("xx_use_lg")
            self.nlp = spacy.load("es_core_news_lg")
        self.version = version
        # STANZA (más lento, resultados levemente mejores)

        # Download the stanza model if necessary
        #stanza.download("es")
        # Initialize the pipeline
        #self.nlp = spacy_stanza.load_pipeline("es",processors="tokenize,pos,lemma,ner")

    def featurize(self, segments):
        """
        Dado una lista de frases retorna una lista de diccionarios con las features de cada frase
        """
        feature_dicts = []
        docs = self.nlp.pipe(segments)
        for doc in docs:
            if self.version == "chq":
                feature_dicts.append(self.get_features_chq(doc))
            elif self.version == "new":
                feature_dicts.append(self.get_features(doc))
            elif self.version == "ff":
                feature_dicts.append(self.get_features_ff(doc))

        return feature_dicts

    def featurize_embs(self, segments):
        return [np.array(doc.vector) for doc in self.nlp_use.pipe(segments)]

    def get_features(self, doc, use_ngrams=True):
        """
        Dado un Doc de Spacy extrae features relevantes en un diccionario
        """
        features = {}
        
        # Entidades
        for ent in doc.ents:
            features[ent.text.lower()] = True
            features[ent.label_] = True
            
        # N-grams de etiquetas POS
        if use_ngrams:
            pos_tags = [t.pos_ for t in doc]
            sentence_ngrams = ngrams(pos_tags,3)
            for ngram in sentence_ngrams:
                features[str(ngram)] = True
        
        # Análisis de tokens
        for token in doc:
            # Remove stopwords
            #if token.is_stop:
            #    continue
            # Morphology: tense, person, mood, etc
            for key,value in token.morph.to_dict().items():
                features[key+value] = True

            # Lemmas + shapes
            if "d" in token.shape_:
                # SI es año agrego un feature de acuerdo a si ya paso o es futuro
                
                if token.shape_ == "dddd":
                    value = int(token.text)
                    if value > 1950 and value < 2050:
                        if value < today.year:
                            features["PAST_YEAR"] = True
                        elif value > today.year:
                            features["COMING_YEAR"] = True
                        else:
                            features["THIS_YEAR"] = True
                
                # Siempre agrego el shape igual
                features[token.shape_] = True
            else:
                features[token.lemma_.lower()] = True

            # POS
            features[token.pos_] = True     
            # Dep
            features[token.dep_] = 1
            # Text
            features[token.text.lower()] = True
            
        #import pdb; pdb.set_trace()
        
        return features

    def get_features_ff(self, doc, use_ngrams=True):
        features = {}

        for tagged_word in doc:
            features[tagged_word.pos_] = True

        # Entidades
        for ent in doc.ents:
            features[ent.text.lower()] = True
            features[ent.label_] = True

        return features


    def get_features_chq(self, doc, use_ngrams=True):
        features = {}

        for tagged_word in doc:
            #pos, lemma, text, tag, dep ,is_punct, like_num, tense
            if tagged_word.is_punct and tagged_word.lemma_ not in "%¿?":
                continue

            features[tagged_word.pos_] = True
            features[tagged_word.lemma_] = True
            features[tagged_word.dep_] = True
            features[tagged_word.text.lower()] = True

            if is_int(tagged_word.lemma_):
                number_of_digits = len(str(tagged_word.lemma_))
                features['%s_digits' %number_of_digits] = True

        if use_ngrams:        
            ctags_chain = [e.pos_ for e in doc]
            ngs = ngrams(ctags_chain, 3)
            for ng in ngs:
                features[str(ng)] = True

        return features

class SpacyFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)

class SpacyFeatureTransformerChq(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer("chq")

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)


class SpacyFeatureTransformerFF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer("ff")

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)

class EmbeddingsFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize_embs(data)