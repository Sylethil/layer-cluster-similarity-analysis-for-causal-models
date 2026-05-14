from nltk.corpus.reader import bnc
import csv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
SIMLEX_999_PATH = str(_PROJECT_ROOT / 'data/SimLex-999/SimLex-999.txt')
MEN_PATH = str(_PROJECT_ROOT / 'data/MEN/EN-MEN-LEM.txt')

def get_bnc():
    bnc_root = str(_PROJECT_ROOT / 'data/BNC/Texts/')
    bnc_reader = bnc.BNCCorpusReader(root=bnc_root, fileids=r'[A-K]/\w*/\w*\.xml')
    return bnc_reader

def randomly(seq, pseudo=True):
    import random
    shuffled = list(seq)  
    if pseudo:
        seed = lambda : 0.479032895084095295148903189394529083928435389203890819038471
        random.shuffle(shuffled, seed)
    else:
        print("shuffling indexes")
        random.shuffle(shuffled) 
        print("done shuffling")
    return list(shuffled)

def bnc_length(pathname='../data/bnc_length.txt'):
    try:
        with open(pathname, 'r') as fh:
            count = int(fh.read())
            return count
    except:
        print("BNC not yet indexed. Calculating length and writing to 'data/count_of_bnc_sentences.txt'")
        bnc_reader = get_bnc()
        corpus = bnc_reader.tagged_sents(strip_space=True)
        length = len(corpus)
        with open(pathname, 'w') as disk:
            disk.write(str(length))
        return length

def bnc_sentence_to_string(sentence):
    words = [word.lower() for (word, pos) in sentence]
    return " ".join(words)

def get_simlex999(path=SIMLEX_999_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word1", "word2", "POS", "SimLex999", "conc_w1", "conc_w2", "concQ", "assoc_USF", "sim_assoc333", "SD_simlex"])
        line_count = 0
        headers = next(reader)
        for row in reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['similarity'] = float(row['SimLex999'])
            row['conc_w1'] = float(row['conc_w1'])
            row['conc_w2'] = float(row['conc_w2'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from simlex999 dataset" % line_count)
    return data

def get_men(path=MEN_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=["word1", "word2", "relatedness"])
        line_count = 0
        for row in csv_reader:
            # remove the POS tag at the end of the word
            row['word1'] = row['word1'].lower()[:-2]
            row['word2'] = row['word2'].lower()[:-2]
            row['relatedness'] = float(row['relatedness'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from MEN relatedness dataset" % line_count)
    return data 
