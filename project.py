import re
import json
from math import log
from collections import Counter

global idfDict
global list_tf


def tokenizer(text):
    return text.lower()


def set_list(my_list):
    dict_list=dict(Counter(my_list))
    return dict_list

def tfidf_paragrafs():

    list_idf_paragraf=[]
    list_tf_paragraf=[]
    list_idf_doc=[]
    list_tf_doc=[]
    N=counter_paragraf()
    list_document=open_json()
    for i in list_document[0] :
     with open(f'..\data\\document_{i}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         patern_split_words = "[,. :;|]+"
     for i in range(len(my_list)):
           my_list1 = re.split(patern_split_words, my_list[i])
           dict_list1=set_list(my_list1)
           list_idf_paragraf.append(dict_list1)
           list_tf_paragraf.append(TF_compute(dict_list1,dict_list1))
     list_idf_doc.append(IDF_compute(list_idf_paragraf,N))
     list_tf_doc.append(list_tf_paragraf)
    return(list_idf_doc,list_tf_doc)

def counter_paragraf():
    count=0
    list_document=open_json()
    for i in list_document[0] :
      with open(f'..\data\\document_{i}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())    
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         count+=len(my_list)
    return(count)
def open_json():
    with open('..\data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)
        documents=new_data[0]["candidate_documents_id"]
        query=new_data[0]["query"]
    return([documents,query])


def TF_compute(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)


def IDF_compute(docList,N):
    global idfDict
    idfDict = {}
    list_help=[]
    for i in range(len(docList)):
     idfDict = dict.fromkeys(docList[i].keys(), 0)
     for word, val in docList[i].items():
        idfDict[word] = log((N+1)/ (float(val) + 1))
        
     list_help.append(idfDict)
    return(list_help)


# idf_paragraf=tfidf_paragraf()[0]
# tf_paragraf=tfidf_paragraf()[1]
# dict_tfidf={}
# tf_idf_list_paragrafs=[]
# for i in range(len(tf_paragraf)):
#     for j in tf_paragraf[i]:
#       dict_tfidf[j]=tf_paragraf[i][j]*idf_paragraf[i][j]
#     tf_idf_list_paragrafs.append(dict_tfidf)
#     dict_tfidf={}

# print(tf_idf_list_paragrafs)


print(tfidf_paragrafs()[0][0][0])



