import re
import json
from math import log
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
import string
import copy

global idfDict
global list_tf
global list_document_query

def tokenizer(text):
    return text.lower()


def set_list(my_list):
    dict_list=dict(Counter(my_list))
    return dict_list

def tfidf_paragrafs():
    global querylist
    querylist= re.split("[,. :;|]+", open_json()[1])
    list_tfidf_paragraf=[]
    list_tfidf_doc=[]
    N=counter_paragraf()[0]
    tf_idf_query=TFIdf_compute(set_list(re.split("[,. :;|]+", open_json()[1])),N,querylist)
    list_document=open_json()
    for i in list_document[0] :
     with open(f'..\data\\document_{i}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         patern_split_words = "[,. :;|]+"
     for j in range(len(my_list)):
           my_list1 = re.split(patern_split_words, my_list[j])
           dict_list1=set_list(my_list1)
           list_tfidf_paragraf.append(TFIdf_compute(dict_list1,N,querylist))
     list_tfidf_doc.append(list_tfidf_paragraf)
     list_tfidf_paragraf=[]
    return([list_tfidf_doc,tf_idf_query])

def counter_paragraf():
    count=0
    list_len_paragraf_each_doc=[]
    list_document_query=open_json()
    for i in list_document_query[0] :
      with open(f'..\data\\document_{i}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())    
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         count+=len(my_list)
         list_len_paragraf_each_doc.append(count)
    return([count,list_len_paragraf_each_doc])

def open_json():
    with open('..\data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)
        documents=new_data[18]["candidate_documents_id"]
        query=new_data[18]["query"]
    return([documents,query.lower()])



def TFIdf_compute(wordDict,N,querylist):
    
    tfidfDict = {}
    
    corpusCount = len(wordDict)
    for i in querylist :
        if i in wordDict:
          tfidfDict[i] = (wordDict[i]/float(corpusCount))*(log((N+1)/ (float(wordDict[i]) + 1)))
    return(tfidfDict)


def tf_idf_docANDquery():
    
    
    list_tf_idf_doc=[]
    querylist=re.split("[,. :;|]+", open_json()[1])
    dict_querylist={}
    for i in querylist :
       dict_querylist[i]=0
    dictionary=copy.copy(dict_querylist)
    dictionary2=copy.copy(dict_querylist)
    tf_idf=tfidf_paragrafs()
    tf_idf_parags=tf_idf[0]
    tf_idf_query=tf_idf[1]
    tfidf_each_paragraf=[]
    for i in range(len(tf_idf_parags)) :
        for j in range(len(tf_idf_parags[i])):
            for k in tf_idf_parags[i][j]:
                if k in dictionary :
                    dictionary[k]+=tf_idf_parags[i][j][k]
                if k in dictionary2:
                    dictionary2[k]+=tf_idf_parags[i][j][k]
            tfidf_each_paragraf.append(dictionary2)
            dictionary2=copy.copy(dict_querylist)
        list_tf_idf_doc.append(dictionary)
        dictionary=copy.copy(dict_querylist)
    return [list_tf_idf_doc,list(tf_idf_query.values()),tfidf_each_paragraf]


def cos():
    list_cosine=[]
    list_cosine2=[]
    list_select_doc=[]
    tf_idf_doc1=tf_idf_docANDquery()
    for i in tf_idf_doc1[0]:
        list_cosine.append(cosine_similarity(list(i.values()),tf_idf_doc1[1]))
    max_list=max(list_cosine)
    for i in range(len(list_cosine)):
        if(list_cosine[i]==max_list):
            print(open_json()[0][i])
    # find selected paragraf
    for i in tf_idf_doc1[2]:
        list_cosine2.append(cosine_similarity(list(i.values()),tf_idf_doc1[1]))      
    max_list2=max(list_cosine2)
    z=0
    list_is_selected=[]
    for i in range(len(list_cosine2)):
        if(list_cosine2[i]==max_list2 and z==0):
            for j in range(len(counter_paragraf()[1])):
                if int(counter_paragraf()[1][j])>i :
                    print(i-int(counter_paragraf()[1][j-1]))
                    z=1
    



def cosine_similarity(v1, v2):
    """
    v1: first vector
    v2: second vector
    return: cosine similarity between v1 and v2
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if((norm_v1 * norm_v2) !=0):
       return  dot_product / (norm_v1 * norm_v2)
    else :
        return(0)


cos()