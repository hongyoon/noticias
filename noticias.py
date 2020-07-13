import requests, time, locale, spacy, logging, string, os, textdistance

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from datetime import date
from google.cloud import translate

import config, config_google

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

class NewsCrawler():

    def __init__(self):
        self.base_url = config.BASE_URL
        self.logger = logging.getLogger(__name__)
        

    def getNewsUrls(self): # get news url
        req = requests.get(self.base_url)
        coverpage = req.content

        soup = BeautifulSoup(coverpage, 'html5lib')
        links = soup.find_all('a',href=True)

        urls = []
        for link in links:
            if link['href'].startswith('/mundo/noticias'):
                _url = 'https://www.bbc.com'+ link['href']
                urls.append(_url)

        return urls

    def getNewsText(self, soup): # get news text
        ret = []
        for p in soup.find_all('p', attrs={'aria-hidden':None, 'class':None}):
            ret.append(p.getText())

        return ' '.join(ret)

    def getNewsData(self, url): # get news url data
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html5lib')
        fetcha = soup.find('div', attrs={'class':"date date--v2"})
        if not fetcha:
            fetcha = soup.find('time') 
        date = time.strptime(fetcha.getText(), "%d %B %Y")
        text = self.getNewsText(soup)

        return {'url':url, 'date':date, 'text': text}

    def getPreviousUrls(self):
        df_url = pd.read_csv(config.PATH_DB_URL, sep='\t', index_col=None)
        return set(df_url['url'])

    def getListNewsData(self):
        newsData = []
        urls = self.getNewsUrls()
        today = date.today()
        
        # ha venido antes?
        previous_url = self.getPreviousUrls()
        target_urls = list(filter(lambda x: x not in previous_url, urls))

        listNewsData = []
        logging.warning(target_urls)
        for url in target_urls:
            newsData = self.getNewsData(url)
            if newsData['text']:
                logging.warning(url)
                listNewsData.append(newsData)

                ################## REMOVE THIS IN PRODUCTION
                if len(listNewsData) == 2:
                    break

        return listNewsData

class WordDistiller():
    # get raw text data
    # 1. tokenize
    # 2. process new words
    # 3. remove cognate
    # 4. tf-idf 

    def __init__(self):
        self.nlp = spacy.load('es_core_news_sm')
        self.wordDict = self.getWordDict()

        # list of tokens for tf-idf
        self.listTokens = []

        # super set of all words
        self.vocab = set([])

        # new words
        self.newWords = []

        # interesting words
        self.interestingWords = set()

        # for genAnki
        self.palabra2frases = {}

    def getInterestingWords(self, listNewsData):
        
        # tokenize to get vocab, listTokens, palabra2frases
        self.tokenizeNewsData(listNewsData)

        # get new word
        logging.warning("getting new word")
        self.getNewWords()
        logging.warning(len(self.newWords))

        logging.warning("length of word dict is {}".format(len(self.wordDict)))
        # update WordDict
        self.updateWordDict()
        logging.warning("length of word dict is {}".format(len(self.wordDict)))

        # get df_newWord
        if len(self.newWords) > 0:
            # update df_words
            df_newWords = self.getDfNewWords()
            df_word = pd.read_csv(config.PATH_DB_WORD, sep='\t', index_col=None)
            df_merged = df_word.append(df_newWords)
            df_merged.to_csv(config.PATH_DB_WORD, sep='\t', index=False)
        else: # if no new word (eventually it will happen)
            df_newWords = df_word

        # get cognate
        scores= df_word.score
        cognate_threshold = np.percentile(scores,config.COGNATE_PERCENTILE)
        cognates = df_newWords.loc[df_newWords['score'] >= cognate_threshold]['src'].values

        # creating tf idf pipeline
        corpus = [' '.join(token) for token in self.listTokens]
        pipe = self.get_tf_idf_pipeline(stop_words= cognates, corpus=corpus, new_word_only = config.NEW_WORD_ONLY)

        feature_names = np.array(pipe['count'].get_feature_name())
        tf_idf_vector=pipe['tfidf'].transform(pipe['count'].transform(corpus))

        num_row, _ = tf_idf_vector.shape

        for idx in range(num_row):
            arr = tf_idf_vector.getrow(idx).toarray()
            idxs = arr[0].argsort()[-20:][::-1]
            self.interestingWords.update(set(feature_names[idxs]))
    



    def is_valid_token(self,token):
        if token.text.isdigit():
            return False

        if token.text in config.STOP_WORDS:
            return False

        if token.pos_ in config.STOP_POS:
            return False
 
        return True

    def is_valid_sentence(self, sentence):
        return sentence not in config.STOP_SENTENCES

    def getTokenizeText(self, text):
        tokens = []
        # parse text
        doc = self.nlp(text)

        # iterate over span (sentence + metadata)
        for span in doc.sents:
            sentence = span.text
            if self.is_valid_sentence(sentence):
                for token in span:
                    if self.is_valid_token(token):
                        # add to words
                        t = token.text.lower()
                        tokens.append(t)

                        # update t-sentence dictionary
                        frases = self.palabra2frases.get(t, [])
                        frases.append(sentence)
                        self.palabra2frases[t] = frases
        return tokens

    def tokenizeNewsData(self, listNewsData):
        for newData in listNewsData:
            text = newData['text'].translate(str.maketrans('', '', string.punctuation))
            tokens =self.getTokenizeText(text)
            self.listTokens.append(tokens)
            self.vocab.update(set(tokens))

    def getWordDict(self):
        df_word = pd.read_csv(config.PATH_DB_WORD, sep='\t', index_col=None)
        return dict(zip(list(df_word['src']),list(df_word['tran'])))

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def getNewWords(self):
        for v in self.vocab:
            if v not in self.wordDict.keys():
                self.newWords.append(v)

    def updateWordDict(self):
        if self.newWords: # if newWord is not empty
            # initialize google translation service
            client = translate.TranslationServiceClient()
            parent = client.location_path(config_google.PROJECT_ID, 'global')
            
            newWordsDict = {}
            for c in self.chunks(self.newWords, 1024):        
                response = client.translate_text(
                    contents=c,
                    target_language_code='en',
                    source_language_code='es',
                    parent=parent,
                )

                newWordsDict.update(dict(zip(c,[t.translated_text for t in response.translations])))
            
            self.wordDict.update(newWordsDict)

    def getDfNewWords(self):
        trans = [self.wordDict[w] for w in self.newWords]
        listDict = []
        for w, t in zip(self.newWords, trans):
            sim = textdistance.jaro_winkler.normalized_similarity(w,t)
            row ={'src':w, 'tran': t, 'score':sim}
            listDict.append(row)
        df_newWords = pd.DataFrame(listDict)
        return df_newWords

    def get_tf_idf_pipeline(self, stop_words = None, corpus = None, new_word_only=False):
        if new_word_only:
            new_old_word = list(self.wordDict.keys())
            old_word = list(filter(lambda x: x not in self.newWords, new_old_word))

        stop_words = stop_words + old_word

        pipe = Pipeline([('count', CountVectorizer(stop_words=stop_words)),('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True))]).fit(corpus)
        return pipe


# initialize google credential
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_google.PATH_CREDENTIAL
os.environ['PROJECT_ID'] = config_google.PROJECT_ID

# set locale to Spanish
locale.setlocale(locale.LC_TIME, config.LOCALE_ES)

# crawling
nc = NewsCrawler()
listNewsData = nc.getListNewsData()
del nc

# distilling interesting words
wd = WordDistiller()
wd.getInterestingWords([listNewsData[0],listNewsData[1]])

print(wd.interestingWords)
print(wd.newWords)




    

    
