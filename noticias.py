import requests, locale, spacy, logging, string, os, textdistance, pickle, random, datetime

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from google.cloud import translate

import config, config_google

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import genanki

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class NewsCrawler():

    def __init__(self):
        self.base_url = config.BASE_URL
        self.logger = logging.getLogger(__name__)
        self.df_url = self.loadUrlDb()
        self.today = datetime.date.today()

    def loadUrlDb(self):
        return pd.read_csv(config.PATH_DB_URL, sep='\t', index_col=None)

    def updateUrlDb(self, df_temp):
        df_merged = self.df_url.append(df_temp)
        df_merged.to_csv(config.PATH_DB_URL, sep='\t', index=False)
        self.df_url = df_merged

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
        date_obj = datetime.datetime.strptime(fetcha.getText(), "%d %B %Y")
        text = self.getNewsText(soup)

        return {'url':url, 'date':date_obj.strftime('%Y-%m-%d'), 'text': text}

    def saveRawData(self, listNewsData):
        str_today = self.today.strftime('%Y-%m-%d')
        df_raw_data = pd.DataFrame(listNewsData)
        df_raw_data.to_csv(config.PATH_RAW_DATA_FOLDER + "{}.tsv".format(str_today) , sep='\t', index=False)
        

    def getListNewsData(self):
        # print list news data; update db_url
        newsData = []
        urls = self.getNewsUrls()
        
        # ha venido antes?
        previous_urls = set(self.df_url['url'])
        target_urls = list(filter(lambda x: x not in previous_urls, urls))

        listNewsData = []
        logging.warning(target_urls)
        for url in target_urls:
            newsData = self.getNewsData(url)
            if newsData['text']:
                logging.warning(url)
                listNewsData.append(newsData)

                ################# REMOVE THIS IN PRODUCTION
                #if len(listNewsData) == 2:
                #    break
        
        
        temp = []
        str_today = self.today.strftime('%Y-%m-%d')
        for newsData in listNewsData:
            temp.append({'url':newsData['url'], 'crawl_date':str_today})

        df_temp = pd.DataFrame(temp)
        self.updateUrlDb(df_temp)
        self.saveRawData(listNewsData)

        return listNewsData

class WordDistiller():
    # get raw text data
    # 1. tokenize
    # 2. process new words
    # 3. remove cognate
    # 4. find interesting words
    # 5. update interesting words 

    def __init__(self):
        self.nlp = spacy.load('es_core_news_sm')
        self.df_word = self.loadWordDb()
        logging.warning(len(self.df_word))
        self.df_interesting_word = self.loadInterestingWordDb()
        self.wordDict = self.getWordDict()

        # list of tokens for tf-idf
        self.listTokens = []

        # super set of all words
        self.vocab = set([])

        # for genAnki
        self.palabra2frases = {}

    def loadWordDb(self):
        df = pd.read_csv(config.PATH_DB_WORD, sep='\t', index_col=None)
        return df

    def updateWordDb(self, df_temp):
        df_merged = self.df_word.append(df_temp)
        df_merged.to_csv(config.PATH_DB_WORD, sep='\t', index=False)
        self.df_word = df_merged
        self.wordDict = self.getWordDict()

    def loadInterestingWordDb(self):
        df = pd.read_csv(config.PATH_DB_INTERESTING_WORD, sep='\t', index_col=None)
        return df 

    def updateInterestingWordDb(self, df_temp):
        df_merged = self.df_interesting_word.append(df_temp)
        df_merged.to_csv(config.PATH_DB_INTERESTING_WORD, sep='\t', index=False)
        self.df_interesting_word = df_merged

    def getWordDict(self):
        return dict(zip(list(self.df_word['src']),list(self.df_word['tran'])))

    def createAnkiDeck(self, listNewsData):
        logging.warning('time to get some interesting words')
        dict_iws = self.getInterestingWords(listNewsData)

        id_deck = random.randrange(1 << 30, 1 << 31)
        name_deck = self.today.strftime('%Y-%m-%d')
        deck = genanki.Deck(id_deck,name_deck)

        for dict_iw in dict_iws:
            deck.add_note(genanki.Note(model=config.MODEL, fields=[dict_iw['src'], dict_iw['tran'], dict_iw['sentence']]))
        
        filename = f"{config.PATH_DECKS_FOLDER}{name_deck}.apkg"
        logging.warning(filename)
        genanki.Package(deck).write_to_file(filename)

        return filename

    def getInterestingWords(self, listNewsData):
        # tokenize to get vocab, listTokens, palabra2frases
        self.tokenizeNewsData(listNewsData)

        df_newWords = self.getDfNewWords()
        # get df_newWord
        if df_newWords is not None:
            # update df_words
            df_newWords = self.getDfNewWords()
            self.updateWordDb(df_newWords)


        # get cognate
        logging.warning("getting cognate")
        scores= self.df_word.score
        cognate_threshold = np.percentile(scores,config.COGNATE_PERCENTILE)
        idx_cognate = self.df_word['score'] >= cognate_threshold
        cognates = self.df_word.loc[idx_cognate]['src'].values
        logging.warning(len(cognates))

        # creating tf idf pipeline
        logging.warning("making tf-idf")
        corpus = [' '.join(token) for token in self.listTokens]
        pipe = self.get_tf_idf_pipeline(stop_words= cognates, corpus=corpus, new_words_only = config.NEW_WORDS_ONLY)
        pipe

        feature_names = np.array(pipe['count'].get_feature_names())
        tf_idf_vector=pipe['tfidf'].transform(pipe['count'].transform(corpus))

        # get interesting words
        num_row, _ = tf_idf_vector.shape

        interestingWords = set()

        for idx in range(num_row):
            arr = tf_idf_vector.getrow(idx).toarray()
            idxs = arr[0].argsort()[-config.NUM_INTERESTING_WORDS:][::-1]
            interestingWords.update(set(feature_names[idxs]))

        #Update InterestingWordDb
        list_dict = []
        for iw in interestingWords:
            sentence = random.choice(self.palabra2frases.get(iw, None))
            a_dict = {'src':iw,'tran':self.wordDict[iw],'sentence':sentence}
            list_dict.append(a_dict)

        df_temp = pd.DataFrame(list_dict)
        self.updateInterestingWordDb(df_temp)

        return list_dict

    ### 1. let's tokenize ###
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
                cleanText = sentence.translate(str.maketrans('', '', string.punctuation))
                doc_sentence = self.nlp(cleanText)
                for token in doc_sentence:
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
            # text = newData['text'].translate(str.maketrans('', '', string.punctuation))
            text = newData['text']
            tokens =self.getTokenizeText(text)
            self.listTokens.append(tokens)
            self.vocab.update(set(tokens))

    ### 2. Process new words ###
    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def getDfNewWords(self):
        newWords = []
        for v in self.vocab:
            if v not in self.wordDict.keys():
                newWords.append(v)
        
        # if there are new words using google translate api
        if newWords:
            client = translate.TranslationServiceClient()
            parent = client.location_path(config_google.PROJECT_ID,'global')

            newWordsDict = {}
            for c in self.chunks(newWords, 1024):
                r = client.translate_text(
                    contents=c,
                    target_language_code = 'en',
                    source_language_code = 'es',
                    parent = parent,
                )

                newWordsDict.update(dict(zip(c,[t.translated_text for t in r.translations])))

            listDict = []
            for s, t in zip(list(newWordsDict.keys()), list(newWordsDict.values())):
                sim = textdistance.jaro_winkler.normalized_similarity(s.lower(),t.lower())
                row = {'src':s, 'tran':t, 'score':sim} 
                listDict.append(row)
            df_newWords = pd.DataFrame(listDict)
            return df_newWords
        else: 
            return None

    def get_tf_idf_pipeline(self, stop_words = None, corpus = None, new_words_only=False):
        if new_words_only:
            df_previous_iws = self.df_interesting_word
            old_words = df_previous_iws['src'].values
            stop_words = np.concatenate([stop_words,old_words])

        pipe = Pipeline([('count', CountVectorizer(stop_words=list(stop_words))),('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True))]).fit(corpus)
        return pipe

def uploadAnkiFile(filepath):
    GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = "config\\drive-client-secrets.json"
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)
    
    file_drive = drive.CreateFile({'parents':[{'id': config_google.DRIVE_ANKI_FOLDER_ID}], 'title':os.path.basename(filepath) })  
    file_drive.SetContentFile(filepath) 
    file_drive.Upload()
    return


# initialize google credential
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_google.PATH_CREDENTIAL
os.environ['PROJECT_ID'] = config_google.PROJECT_ID

# set locale to Spanish
locale.setlocale(locale.LC_TIME, config.LOCALE_ES)

# # crawling
nc = NewsCrawler()
listNewsData = nc.getListNewsData()
del nc

# with open('newsData.pk','wb') as f:
#     pickle.dump(listNewsData,f)

# with open('newsData.pk','rb') as f:
#     listNewsData = pickle.load(f)

# distilling interesting words
# wd = WordDistiller()
# wd.createAnkiDeck(listNewsData)
# del wd

# uploadAnkiFile("decks\\2020-07-16.apkg")



    

    
