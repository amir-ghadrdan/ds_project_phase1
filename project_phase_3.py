import re
from math import log
from collections import Counter
from difflib import SequenceMatcher
from matplotlib import pyplot
import numpy as np
import copy
import heapq
import string
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


global idfDict
global list_tf
global list_document_query
global inputs_doc_query


def set_list(my_list):
    dict_list = dict(Counter(my_list))
    return dict_list


def tokenizer(text):
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stop_words]
    return filtered_tokens


def tfidf_paragraf_docs():
    global inputs_doc_query
    global querylist
    global list_document_query
    list_document_query = inputs_doc_query
    querylist = re.split("[,. :;|]+", list_document_query[1])
    list_tfidf_paragraf = []
    list_tfidf_doc = []
    N = counter_paragraf()[0]
    tf_idf_query = TFIdf_compute_and_use_SequenceMatcher(set_list(re.split("[,. :;|]+", list_document_query[1])), N, querylist)
    for i in list_document_query[0]:
        with open(f'data\\document_{i}.txt', "r", encoding="utf-8") as doc:
            string = doc.read().lower()
            # split each paragraf by \n
            patern_split_paragraf = "[\n]+"
            my_list = re.split(patern_split_paragraf, string)
            patern_split_words = "[,. :;|]+"
        for j in range(len(my_list)):
            my_list1 = re.split(patern_split_words, my_list[j])
            dict_list1 = set_list(my_list1)
            list_tfidf_paragraf.append(TFIdf_compute_and_use_SequenceMatcher(dict_list1, N, querylist))
        list_tfidf_doc.append(list_tfidf_paragraf)
        list_tfidf_paragraf = []
    return ([list_tfidf_doc, tf_idf_query])


def counter_paragraf():
    global list_document_query, inputs_doc_query
    count = 0
    list_len_paragraf_each_doc = []
    list_document_query = inputs_doc_query
    for i in list_document_query[0]:
        with open(f'data\\document_{i}.txt', "r", encoding="utf-8") as doc:
            string = doc.read().lower()
            patern_split_paragraf = "[\n]+"
            my_list = re.split(patern_split_paragraf, string)
            count += len(my_list)
            list_len_paragraf_each_doc.append(count)
    return ([count, list_len_paragraf_each_doc])


def TFIdf_compute_and_use_SequenceMatcher(wordDict, N, querylist):

    tfidfDict = {}

    corpusCount = len(wordDict)
    for i in querylist:
        for j in wordDict:
            if SequenceMatcher(None, i, j).ratio() > 0.9:
                tfidfDict[i] = (wordDict[j]/float(corpusCount)) * (log((N+1) / (float(wordDict[j]) + 1)))
    return (tfidfDict)


def tf_idf_docANDquery():

    global inputs_doc_query

    list_tf_idf_doc = []
    querylist = re.split("[,. :;|]+", inputs_doc_query[1])
    dict_querylist = {}
    for i in querylist:
        dict_querylist[i] = 0
    ordered_dict_based_query = copy.copy(dict_querylist)
    ordered_dict_based_query2 = copy.copy(dict_querylist)
    tf_idf = tfidf_paragraf_docs()
    tf_idf_parags = tf_idf[0]
    tf_idf_query = tf_idf[1]
    tfidf_each_paragraf = []
    for i in range(len(tf_idf_parags)):
        for j in range(len(tf_idf_parags[i])):
            for k in tf_idf_parags[i][j]:
                if k in ordered_dict_based_query:
                    ordered_dict_based_query[k] += tf_idf_parags[i][j][k]
                if k in ordered_dict_based_query2:
                    ordered_dict_based_query2[k] += tf_idf_parags[i][j][k]
            tfidf_each_paragraf.append(ordered_dict_based_query2)
            ordered_dict_based_query2 = copy.copy(dict_querylist)
        list_tf_idf_doc.append(ordered_dict_based_query)
        ordered_dict_based_query = copy.copy(dict_querylist)
    return [list_tf_idf_doc, list(tf_idf_query.values()), tfidf_each_paragraf]


def find_bestDOC_bestPARAGRAF():
    global inputs_doc_query
    # selected_best_doc
    list_vector_cos_docANDquery = []
    list_vector_cos_paragrafs_selected_doc = []
    value_tf_idf_each_doc = tf_idf_docANDquery()
    for i in value_tf_idf_each_doc[0]:
        list_vector_cos_docANDquery.append(cosine_similarity(list(i.values()), value_tf_idf_each_doc[1]))
    max_vector_docs = max(list_vector_cos_docANDquery)
    for i in range(len(list_vector_cos_docANDquery)):
        if (list_vector_cos_docANDquery[i] == max_vector_docs):
            print(inputs_doc_query[0][i])

    # selected best paragraf

    for i in value_tf_idf_each_doc[2]:
        list_vector_cos_paragrafs_selected_doc.append(cosine_similarity(list(i.values()), value_tf_idf_each_doc[1]))
    max_vector_paragrafs = max(list_vector_cos_paragrafs_selected_doc)
    z = 0
    list_paragraf_best_doc = []
    count_pr = counter_paragraf()[1]
    for i in range(len(list_vector_cos_paragrafs_selected_doc)):
        if (list_vector_cos_paragrafs_selected_doc[i] == max_vector_paragrafs and z == 0):
            for j in range(len(count_pr)):
                if int(count_pr[j]) > i and z == 0:
                    selected_paragraf = i-int(count_pr[j-1])
                    len_list_paragraf_best_doc = int(count_pr[j])-int(count_pr[j-1])
                    z = 1

    for i in range(len_list_paragraf_best_doc):
        if i != selected_paragraf:
            list_paragraf_best_doc.append("0")
        else:
            list_paragraf_best_doc.append("1")

    print(list_paragraf_best_doc)


def cosine_similarity(v1, v2):
    """
    v1: first vector
    v2: second vector
    return: cosine similarity between v1 and v2
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if ((norm_v1 * norm_v2) != 0):
        return dot_product / (norm_v1 * norm_v2)
    else:
        return (0)


def find_most_frequent_word(docuemnt_id):
    with open(f'data\\document_{docuemnt_id}.txt', "r", encoding="utf-8") as doc:
        string = doc.read().lower()
        list_word_in_document = re.split("[,. :;|\n]+", string)
        dict_word = set_list(list_word_in_document)
        max = 0
        for i in dict_word:
            if (dict_word[i] > max):
                max = dict_word[i]
        for key, val in dict_word.items():
            if max == val:
                return key


def find_the_5_important_words(document_id):
    with open(f'data\\document_{document_id}.txt', "r", encoding="utf-8") as doc:
        string = doc.read().lower()
        patern_split_paragraf = "[\n]+"
        tfidfDict = {}
        my_list = re.split(patern_split_paragraf, string)
        for j in range(len(my_list)):
            my_list1 = tokenizer(my_list[j])
            dict_list1 = set_list(my_list1)
            corpusCount = len(dict_list1)
            for i in my_list1:
                if (i in tfidfDict.keys()):
                    tfidfDict[i] += (dict_list1[i]/float(corpusCount)) * (log((len(my_list1)) / (float(dict_list1[i]))))
                else:
                    tfidfDict[i] = (dict_list1[i]/float(corpusCount)) * (log((len(my_list1)) / (float(dict_list1[i]))))
        top_items = heapq.nlargest(5, tfidfDict.items(), key=lambda x: x[1])
        top_keys = [key for key, value in top_items]

        return [tfidfDict, top_keys]


def docs_visualizer(n):
    vector_docs = []
    for i in range(n):
        five_important_words = []
        tfidf_doc, top_keys = find_the_5_important_words(i)[0], find_the_5_important_words(i)[1]
        five_important_words = [tfidf_doc[key] for key in top_keys[:5]]
        vector_docs.append(five_important_words)
    two_dimension_data = PCA(n_components=2).fit_transform(vector_docs)
    pyplot.scatter(two_dimension_data[:, 0], two_dimension_data[:, 1], c=KMeans(n_clusters=4, random_state=0).fit(two_dimension_data).labels_, cmap='plasma')
    pyplot.show()


def input_docANDquery():
    query = input("Write the query: ")
    documents = input("Write the documents: ").split()
    return ([documents, query.lower()])


docs_visualizer(int(input("Please enter a number: ")))