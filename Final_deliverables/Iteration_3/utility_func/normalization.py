# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:25:26 2017

@author: anbarasan.selvarasu
"""

from contractions import CONTRACTION_MAP
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from HTMLParser import HTMLParser
import unicodedata
from pattern.en import tag
from nltk.corpus import wordnet as wn

 
class LinguisticProcessing(object):
    
   
    
                                         
    

    def __init__(self):
        pass
        
        
        
    def tokenize_text(self,text):
        tokens = nltk.word_tokenize(text) 
        tokens = [token.strip() for token in tokens]
        return tokens
    
    def expand_contractions(self,text, contraction_mapping):
        
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
            
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text
        
        
    
    # Annotate text tokens with POS tags
    def pos_tag_text(self,text):
        
        def penn_to_wn_tags(pos_tag):
            if pos_tag.startswith('J'):
                return wn.ADJ
            elif pos_tag.startswith('V'):
                return wn.VERB
            elif pos_tag.startswith('N'):
                return wn.NOUN
            elif pos_tag.startswith('R'):
                return wn.ADV
            else:
                return None
        
        tagged_text = tag(text)
        tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                             for word, pos_tag in
                             tagged_text]
        return tagged_lower_text
        
    # lemmatize text based on POS tags    
    def lemmatize_text(self, text):
        wnl = WordNetLemmatizer()
   
        pos_tagged_text = self.pos_tag_text(text)
        lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                             else word                     
                             for word, pos_tag in pos_tagged_text]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text
        
    
    def remove_special_characters(self,text):
        tokens = self.tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
        
        
    def remove_stopwords(self,text):
        stopword_list = nltk.corpus.stopwords.words('english')
    
        months = ['january','february','march', 'april', 'may', 'june','july','august','september','october','november','december'
                    ,'am','pm'
                    ,'gmt']
        emp_name = []
        greetings = ['hello','hi','thankyou','thanks','regards']
        misc =  ['mr', 'mrs', 'come', 'go', 'get',
                 'tell', 'listen', 'one', 'two', 'three',
                 'four', 'five', 'six', 'seven', 'eight',
                 'nine', 'zero', 'join', 'find', 'make',
                 'say', 'ask', 'tell', 'see', 'try', 'back',
                 'also']
        stopword_list = stopword_list +  months + emp_name + greetings + misc
        tokens = self.tokenize_text(text)
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
    
    def keep_text_characters(self,text):
        filtered_tokens = []
        tokens = self.tokenize_text(text)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
    
    def unescape_html(self,parser, text):
        
        return parser.unescape(text)
    
    def normalize_corpus(self,corpus, lemmatize=True, 
                         only_text_chars=False,
                         tokenize=False):
        #corpus = email_corpus[1]
        html_parser = HTMLParser()
        
        normalized_corpus = []    
        try:
            for text in corpus:
                #text = corpus
                text = html_parser.unescape(text)
                #text = expand_contractions(text, CONTRACTION_MAP)
                if lemmatize:
                    text = self.lemmatize_text(text)
                else:
                    text = text.lower()
                text = self.remove_special_characters(text)
                text = self.remove_stopwords(text)
                if only_text_chars:
                    text = self.keep_text_characters(text)
                
                if tokenize:
                    text = self.tokenize_text(text)
                    normalized_corpus.append(text)
                else:
                    normalized_corpus.append(text)
        except :
           raise ValueError('Error in Normalize Corpus')
             
        return normalized_corpus
    
    
    def parse_document(self,document):
        document = re.sub('\n', ' ', document)
        if isinstance(document, str):
            document = document
        elif isinstance(document, unicode):
            return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
        else:
            raise ValueError('Document is not string or unicode!')
        document = document.strip()
        sentences = nltk.sent_tokenize(document)
        sentences = [sentence.strip() for sentence in sentences]
        return sentences
        
        