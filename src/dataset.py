
import glob
import os
from nltk.tokenize import sent_tokenize

def get_sentences(folder):
    # Load all the tagged sentences included in the .pickle files 
    parsed_sentences = []
    files_path = os.path.join(folder, '*.txt')

    for filename in glob.glob(files_path):
        with open(filename, 'r') as f:
            text = f.read()
            parsed_sentences = parsed_sentences + sent_tokenize(text)

    tagged = get_tagged_sentences(parsed_sentences)
    return tagged

def clean_sentence(sentence):
	return sentence.replace(',','').replace('.','').replace(';','').replace('[','').replace(']','').replace("(Aplausos.)","").replace("(aplausos)","").replace("%"," PERCENTAGE ").replace("$", " money ")        

def get_tagged_sentences(sentences):
	# From a list of sentence, find all the fact-checakble tags in it, else, tag as non fact checkable
	tagged_sentences = []
	for sentence in sentences:
		sentence = clean_sentence(sentence).lower()
		if "<chequeable>" in sentence or "</chequeable>" in sentence:
			tagged_sentences.append({'target': 'fact-checkable', 'sentence': sentence.replace("<chequeable>","").replace("</chequeable>","")})
		else:
			tagged_sentences.append({'target': 'non-fact-checkable', 'sentence': sentence})
			
	return tagged_sentences
