import spacy
#import stanza
#import spacy_stanza
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

# Today reference date
today = datetime.today()

class SpacyFeaturizer:
    def __init__(self):
        # Spacy
        self.nlp = spacy.load("es_core_news_lg")
        
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
            feature_dicts.append(self.get_features(doc))
        return feature_dicts

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
            #features[token.pos_] = True            
            # Dep
            #features[token.dep_] = 1
            # Text
            #features[token.text.lower()] = True
            
        
        #import pdb; pdb.set_trace()
        
        return features

class CustomLinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)