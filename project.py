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
global inputs_doc_query



def tokenizer(text):
    return text.lower()


def set_list(my_list):
    dict_list=dict(Counter(my_list))
    return dict_list

def tfidf_paragraf_docs():
    global inputs_doc_query
    global querylist
    global list_document_query
    list_document_query=inputs_doc_query
    querylist= re.split("[,. :;|]+", list_document_query[1])
    list_tfidf_paragraf=[]
    list_tfidf_doc=[]
    N=counter_paragraf()[0]
    tf_idf_query=TFIdf_compute(set_list(re.split("[,. :;|]+", list_document_query[1])),N,querylist)
    for i in list_document_query[0] :
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
    global list_document_query,inputs_doc_query
    count=0
    list_len_paragraf_each_doc=[]
    list_document_query=inputs_doc_query
    for i in list_document_query[0] :
      with open(f'..\data\\document_{i}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())    
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         count+=len(my_list)
         list_len_paragraf_each_doc.append(count)
    return([count,list_len_paragraf_each_doc])




def TFIdf_compute(wordDict,N,querylist):
    
    tfidfDict = {}
    
    corpusCount = len(wordDict)
    for i in querylist :
        if i in wordDict:
          tfidfDict[i] = (wordDict[i]/float(corpusCount))*(log((N+1)/ (float(wordDict[i]) + 1)))
    return(tfidfDict)


def tf_idf_docANDquery():
    
    global inputs_doc_query

    list_tf_idf_doc=[]
    querylist=re.split("[,. :;|]+", inputs_doc_query[1])
    dict_querylist={}
    for i in querylist :
       dict_querylist[i]=0
    ordered_dict_based_query=copy.copy(dict_querylist)
    ordered_dict_based_query2=copy.copy(dict_querylist)
    tf_idf=tfidf_paragraf_docs()
    tf_idf_parags=tf_idf[0]
    tf_idf_query=tf_idf[1]
    tfidf_each_paragraf=[]
    for i in range(len(tf_idf_parags)) :
        for j in range(len(tf_idf_parags[i])):
            for k in tf_idf_parags[i][j]:
                if k in ordered_dict_based_query :
                    ordered_dict_based_query[k]+=tf_idf_parags[i][j][k]
                if k in ordered_dict_based_query2:
                    ordered_dict_based_query2[k]+=tf_idf_parags[i][j][k]
            tfidf_each_paragraf.append(ordered_dict_based_query2)
            ordered_dict_based_query2=copy.copy(dict_querylist)
        list_tf_idf_doc.append(ordered_dict_based_query)
        ordered_dict_based_query=copy.copy(dict_querylist)
    return [list_tf_idf_doc,list(tf_idf_query.values()),tfidf_each_paragraf]



def find_bestDOC_bestPARAGRAF():
    global inputs_doc_query
    # selected_best_doc
    list_vector_cos_docANDquery=[]
    list_vector_cos_paragrafs_selected_doc=[]
    value_tf_idf_each_doc=tf_idf_docANDquery()
    for i in value_tf_idf_each_doc[0]:
        list_vector_cos_docANDquery.append(cosine_similarity(list(i.values()),value_tf_idf_each_doc[1]))
    max_vector_docs=max(list_vector_cos_docANDquery)
    for i in range(len(list_vector_cos_docANDquery)):
        if(list_vector_cos_docANDquery[i]==max_vector_docs):
            print(inputs_doc_query[0][i])

    # selected best paragraf

    for i in value_tf_idf_each_doc[2]:
        list_vector_cos_paragrafs_selected_doc.append(cosine_similarity(list(i.values()),value_tf_idf_each_doc[1]))  
    max_vector_paragrafs=max(list_vector_cos_paragrafs_selected_doc)
    z=0
    list_paragraf_best_doc=[]
    count_pr=counter_paragraf()[1]
    for i in range(len(list_vector_cos_paragrafs_selected_doc)):
        if(list_vector_cos_paragrafs_selected_doc[i]==max_vector_paragrafs and z==0):
            for j in range(len(count_pr)):
                if int(count_pr[j])>i and z==0:
                    selected_paragraf=i-int(count_pr[j-1])
                    len_list_paragraf_best_doc=int(count_pr[j])-int(count_pr[j-1])
                    z=1
    
    for i in range(len_list_paragraf_best_doc):
        if i !=selected_paragraf :
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
    if((norm_v1 * norm_v2) !=0):
       return  dot_product / (norm_v1 * norm_v2)
    else :
        return(0)


def input_docANDquery():
        query= "An hour to 1 hour 15 minutes."
        documents=[
            11743,
            33098,
            7050,
            24540,
            8638,
            46362,
            20890,
            23059,
            25242,
            12868,
            9284,
            24952,
            36222,
            28056,
            29221,
            6425,
            6463,
            32892,
            9815,
            21
  
        ]
        return([documents,query.lower()])

inputs_doc_query=input_docANDquery()
find_bestDOC_bestPARAGRAF()