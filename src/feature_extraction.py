"""
    Este módulo contiene los componentes para extraer diferentes conjuntos de features.
    En sklearn se los conoce como featurizers o transformers.
"""
import spacy
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np

def is_int(s):
    """
    Función helper para detectar si un string es entero
    """
    try: 
        int(s)
        return True
    except ValueError:
        return False

class SpacyFeaturizer:
    """
    Featurizer basado en el pipeline de Spacy + USE
    """


    def __init__(self, spacy_model="es_core_news_lg",use_model="xx_use_lg"):
        """
        Inicializa el pipeline
        """
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe('universal_sentence_encoder',config={"model_name":use_model})
            

    def featurize(self, frases):
        """
        Dado una lista de frases retorna una lista de diccionarios con las features de cada frase
        Utiliza la función provista como parametro para extraer los features
        """
        feature_dicts = []
        docs = self.nlp.pipe(frases)
        for doc in docs:
                feature_dicts.append(self.get_features_tradicionales(doc))

        return feature_dicts

    def featurize_embs(self, frases):
        """
        Dada una lista de frases retorna lista de embeddings
        """
        return [np.array(doc.vector) for doc in self.nlp.pipe(frases)]

    def get_features_tradicionales(self, doc, use_ngrams=True):
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
            # Morphology: tense, person, mood, etc
            for key,value in token.morph.to_dict().items():
                features[key+value] = True

            # Lemmas + shapes
            if "d" in token.shape_:
                # SI es año agrego un feature de acuerdo a si ya paso o es futuro
                # Today reference date
                today = datetime.today()
                
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

class TraditionalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Featurizer para las features tradicionales. POS, Lemas, Morfología, Entidades, n-gramas
    """
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize(data)

class EmbeddingsFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Featurizer de representaciones vectoriales
    """
    def __init__(self):
        self.featurizer = SpacyFeaturizer()

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.featurizer.featurize_embs(data)