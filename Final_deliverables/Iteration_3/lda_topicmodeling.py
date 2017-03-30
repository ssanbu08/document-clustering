# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:08 2017

@author: anbarasan.selvarasu
"""


from gensim import corpora, models, similarities, matutils
import numpy as np
from utility_func import normalization
from data_preparation import data_preparation as data_prepare
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from lda import lda_topicmodeling_class

print('Load dataset...')
dp = data_prepare.DataPreparation("C:/Senna/Technical/Data Science/Projects/Hands-on/enron_mail_20150507/maildir/")
emails_dict = dp.get_inbox_of()
#print(emails_dict['allen-p'][0][1])

#Initialize the neccessary parameters
emp_emaildocs = []
words = []

lda_topicmodeling_obj = lda_topicmodeling_class.LDATopicModeling()


print('Preprocess dataset...')

new_dict = {}
for employee,mail in list(emails_dict.items()):
    print(employee)
    list_norm_emails = normalize_corpus(mail,only_text_chars=True,lemmatize=True)
    new_dict[employee] = ''.join(list_norm_emails)



# Initialize necessery parameters
num_topics=5
freq_below = 3
freq_above = 0.3


# Generating corpus and id2word
#
print("Generating corpus and id2word...")
norm_tokenized_corpus = normalize_corpus(new_dict.values(), tokenize=True)

import pickle
# pickle.dump(documents, open('processed_inbox_docs_full.pickle', 'wb'))

document_pickle = pickle.load(open('processed_inbox_docs_full.pickle', 'rb'))
tokenized_docs = [i.split() for i in document_pickle.values()]

'''
pickle.dump(tokenized_docs, open('tokenized_docs_lda.pickle', 'wb'))

tokenized_docs = pickle.load(open('tokenized_docs_lda.pickle', 'rb'))
'''


dictionary = corpora.Dictionary(tokenized_docs)
dictionary.filter_extremes(no_below=freq_below, no_above=freq_above)
mapped_corpus = [dictionary.doc2bow(text) 
                 for text in tokenized_docs]

pickle.dump(mapped_corpus, open('mapped_corpus.pickle', 'wb'))
pickle.dump(dictionary, open('dictionary.pickle', 'wb'))

mapped_corpus = pickle.load(open('mapped_corpus.pickle', 'rb'))
dictionary = pickle.load(open('dictionary.pickle', 'rb'))



#  Calculate perplexity with different number of topics_size to analyze
#              
print("Perplexity analysis...")
               
lda_topicmodeling_obj.analyse_num_topics(mapped_corpus,dictionary)



print('Build the model...')
num_topics = 6
lda_gensim = lda_topicmodeling_obj.train_lda_model_gensim(mapped_corpus, dictionary,
                                    total_topics=num_topics)
print(lda_gensim.show_topics())


# Save the model for fast usage
lda_gensim.save('enron_emps_lda_tm_defs_2.pkl')


print('Print the topics...')
lda_topicmodeling_obj.print_topics_gensim(topic_model=lda_gensim,
                    total_topics=num_topics,
                    num_terms=10,
                    display_weights=False) 
                    
    

print("LDA Evaluation...")
lda_topicmodeling_obj.intra_inter(lda_gensim, document_pickle.values(),freq_below,freq_above)


# Wordclouds for each topic
#
print("Topics Wordcloud Visualisation ")
for t in range(2):
    plt.figure()
    li = [list(elem) for elem in lda_gensim.show_topic(0, 200)]
    plt.imshow(WordCloud().fit_words([list(elem) for elem in lda_gensim.show_topic(t, 200)]))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
  




# Comparing similarity between 2 docs by comparing topic vectors
#
print("Comparing similarity between 2 docs by comparing topic vectors")
        
# Creating List for emps_index
emps_index = []
for i, emp_emails in enumerate(tokenized_docs.iteritems()):
    emps_index.append(emp_emails[0])

# find closest docs
employee = 'keavey-p'
print("Find the closest interactive employee to a particular employee")
print("Similar Employee for {} : \t {}" .format(employee,emps_index[lda_topicmodeling_obj.closest_docs(lda_gensim, mapped_corpus ,emps_index.index(employee))]))



# Miscelleaneous analysis
#
print("Miscelleaneous analysis ...")

# topic distribution of each document
topics = [lda_gensim[bow_doc] for bow_doc in mapped_corpus]

# finding most prominent topic from the documents
counts = np.zeros(num_topics)
for doc_topic in topics:
    for t_indx,_ in doc_topic:  
        counts[t_indx] += 1
prom_words = lda_gensim.show_topic(counts.argmax(), 10)
print("Most Prominent Topic : ", counts.argmax())
print(prom_words)


# find average topic size
topics_len = np.array([len(t) for t in topics])
print(np.mean(topics_len))
# find the proportion of docs less than having 2 topics
print(np.mean(topics_len<=2))

