import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spacy

class KeywordExtr():    
    docs = []
    stop_words = set(stopwords.words('english'))
    #stemmer = PorterStemmer() #lemmatizer is used
    lemmatizer = WordNetLemmatizer()
    nlpNER = spacy.load("en_core_web_sm")# for named entity 

    def __init__():
        super().__init__()
        KeywordExtr.ensure_nltk_resources()
    
    @staticmethod
    def ensure_nltk_resources():
        required = [
            'punkt',
            'punkt_tab',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng',
            'words',
            'wordnet',
            'stopwords',
            'omw-1.4',
            'maxent_ne_chunker',
            'maxent_ne_chunker_tab'
        ]
        for resource in required:
            try:
                nltk.data.find(f'{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}...")
                nltk.download(resource)

    
    # POS tag conversion for lemmatizer
    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    @staticmethod
    def preprocess_Seg_Token_POS(doc):
        sentences = sent_tokenize(doc) #Segmentation
        all_words = []
        for sentence in sentences:
            sentence = sentence.lower().replace(".", "") 
            tokens = word_tokenize(sentence) #Tokenize
            tagged = pos_tag(tokens) #POS_PARSING
            all_words.append(tagged)
        return all_words
    
    @staticmethod
    def preprocess_stop_words_lemmatizer(doc):
        all_words = []
        for tagged in doc:
            for word, tag in tagged:
                #Removing Stop Words
                if word.isalpha() and word not in KeywordExtr.stop_words:
                    #STEMMING or lemmating 
                    lemma = KeywordExtr.lemmatizer.lemmatize(word, KeywordExtr.get_wordnet_pos(tag))
                    all_words.append(lemma)
        return all_words
    
    def get_tf_idf(docs):
        seg_Token_POSs = [KeywordExtr.preprocess_Seg_Token_POS(doc) for doc in docs]

        stop_words_lemmatizers = [KeywordExtr.preprocess_stop_words_lemmatizer(doc) for doc in seg_Token_POSs]
        processed_docs = [' '.join(doc) for doc in stop_words_lemmatizers]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=[f'Doc{i+1}' for i in range(len(docs))])
        posArr = np.empty(len(feature_names), dtype=f'<U{len(feature_names)}')
        for i in range(len(posArr)):
            for poss in seg_Token_POSs:
                for pos in poss:
                    for word, tag in pos:
                        if(word == feature_names[i]):
                            posArr[i] = tag

        df['POS'] = posArr
        return df
    
    def get_ner(docs):
        named = [KeywordExtr.nlpNER(doc) for doc in docs]
        data = []
        for i, n in enumerate(named):
            ind = i + 1
            for ent in n.ents:
                data.append({
                    "Doc": f"{ind}",
                    "Text": ent.text,
                    "Label": ent.label_
                })
        
        df = pd.DataFrame(data)
        return df