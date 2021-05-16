#
#	Provee las frases para el entrenamiento de los modelos
#
import glob
import os
import csv
from nltk.tokenize import sent_tokenize

def save_csv(folder, sentences):
	"""
	Dado un conjunto de frases con etiquetas las guarda en formato csv
	"""
	filename = os.path.join(folder, 'frases.csv')

	with open(filename, mode='w') as sentences_file:
		file_writer = csv.writer(sentences_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for sentence in sentences:
			file_writer.writerow([sentence['sentence'],sentence['target']])
	
def get_sentences(folder):
	"""
	Dada la dirección de una carpeta recorre todos los .txt presentes y extrae el texto,
	luego lo parsea en frases y extrae las etiquetas de chequeable/no chequeable de cada frase
	"""
	parsed_sentences = []
	files_path = os.path.join(folder, '*.txt')
	
	for filename in glob.glob(files_path):
		with open(filename, 'r') as f:
			text = f.read()
			sentences_tokenized = sent_tokenize(text)
			sentences_tokenized = [i for s in sentences_tokenized for i in s.split("\n") if len(i) > 1]
			parsed_sentences = parsed_sentences + sentences_tokenized

	tagged = get_tagged_sentences(parsed_sentences)
	return tagged

def get_sentences_csv(folder):
	# Load all the tagged sentences included in the .pickle files 
	filename = os.path.join(folder, 'frases_revisadas.csv')
	
	with open(filename, 'r') as f:
		sentences_reader = csv.reader(f, delimiter=',',quotechar='"')
		sentences = [
			{'target': r[1],
			'sentence': r[0]} for r in sentences_reader
		]

	return sentences

def clean(sentence):
	#return sentence.replace(',','').replace('.','').replace(';','').replace('[','').replace(']','').replace("(Aplausos.)","").replace("(aplausos)","").replace("%"," PERCENTAGE ").replace("$", " money ")        
	return sentence.replace("-","").replace(";","").replace(',','').replace(".","")

def get_tagged_sentences(sentences):
	# From a list of sentence, find all the fact-checakble tags in it, else, tag as non fact checkable
	tagged_sentences = []
	for sentence in sentences:
		sentence = clean(sentence)
		no_tags = sentence.replace("<chequeable>","").replace("</chequeable>","")	
		if "<chequeable>" in sentence or "</chequeable>" in sentence:
			tagged_sentences.append({'target': 'fact-checkable', 'sentence': no_tags})
		else:
			tagged_sentences.append({'target': 'non-fact-checkable', 'sentence': sentence})
	
	FOLDER = os.path.join('..','data','tagged_corpus')

	# save to csv
	save_csv(FOLDER,tagged_sentences)

	return tagged_sentences

if __name__=="__main__":
	FOLDER = os.path.join('..','data','tagged_corpus')

	get_sentences(FOLDER)