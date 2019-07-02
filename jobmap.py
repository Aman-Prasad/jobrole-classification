import re
#import time
import logging
logger = logging.getLogger(__name__)
import pickle
import spacy
from collections import Counter

model_directory = r'C:\Users\user\Desktop\work\jobrole classification\categories\model\NewNer'
nnlp = spacy.load(model_directory)
filename = r'C:\Users\user\Desktop\work\jobrole classification\categories\model\textmap.sav'

def get_desig(text):
    doc_to_test=nnlp(text)
    for ent in doc_to_test.ents:
        if ent.label_ =='designation':
            if ent.text != '':
                return(ent.text)
    


if __name__ == "__main__":
    pick=[]
    filepath =r'C:\Users\user\Desktop\work\jobrole classification\job roles\cfo\resume975.txt'
    with open(filepath,'r') as f:
        z=f.read()
    z = re.sub(r'[^\x00-\x7F]+',' ', z)
    z = re.sub(r'[^\S\n]+', ' ', re.sub(r'\s*[\n\t\r]\s*', '\n', z))
    z=re.sub('’','\'',z)
    z=z.strip('\n')
    x = get_desig(z)
    res,vectorize = pickle.load(open(filename, 'rb'))
    bagOfWords = vectorize.transform([x])
    test = bagOfWords.toarray()
    for i in res:
        pick.append((i.predict(test))[0])     
    votes = Counter(pick)
    print("\n It falls in ",votes.most_common(1)[0][0]," category.\n\n")
    
                