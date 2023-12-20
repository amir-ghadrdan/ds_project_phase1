import re
import json
from math import log
from collections import Counter
from tokenizer import tokenizer

global idfDict
global list_tf
tf_idf = []
dict_paragraf_tf_idf={}

def set_list(my_list):
    return Counter(my_list)


def IDF_Paragraf(num):
    list_help=[]
    #open files document and sent words in each paragraf to idf_compute to calculate idf
    with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         pattern_split_paragraf = "[\n]+"
         my_list = re.split(pattern_split_paragraf, string)
         pattern_split_words = "[,. :;|]+"
    for i in range(len(my_list)):
           my_list1 = re.split(pattern_split_words, my_list[i])
           dict_list1=set_list(my_list1)
           list_help.append(dict_list1)

    return IDF_compute(list_help)


def TF_Paragraf(num):
        global list_tf
        list_tf=[]
        with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         pattern_split_words = "[,. :;|]+"
         for i in range(len(my_list)):
           my_list1 = re.split(pattern_split_words, my_list[i])
           dict_list1=set_list(my_list1)
           list_tf.append(TF_compute(dict_list1,dict_list1))
        return list_tf

def open_json():
    with open('data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)


def TF_compute(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)


def IDF_compute(docList):
    global idfDict
    idfDict = {}
    list_help=[]
    N = len(docList)
    for i in range(N):
     idfDict = dict.fromkeys(docList[i].keys(), 0)
     for word, val in docList[i].items():
        idfDict[word] = log(N/ (float(val) + 1))
        
     list_help.append(idfDict)
    return(list_help)

for i in range(len(IDF_Paragraf(0))):
    for j in IDF_Paragraf(0)[i].keys():
        dict_paragraf_tf_idf[j]=IDF_Paragraf(0)[i][j] * TF_Paragraf(0)[i][j]
    tf_idf.append(dict_paragraf_tf_idf)
    dict_paragraf_tf_idf={}
