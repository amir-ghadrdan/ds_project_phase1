import re
import json
from math import log
from collections import Counter

global idfDict
global list_tf


def tokenizer(text):
    return text.lower()


def set_list(my_list):

    return dict(Counter(my_list))


def calculate_idf_words_each_paragraf(num):
    # with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
    #     string = tokenizer(doc.read())
    list_help=[]
    #open files document and sent words in each paragraf to computeidf to calculate idf
    with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         patern_split_words = "[,. :;|]+"
    for i in range(len(my_list)):
           my_list1 = re.split(patern_split_words, my_list[i])
           dict_list1=set_list(my_list1)
           list_help.append(dict_list1)

    print(computeIDF(list_help))
    # with open(f'file_data\\file_{num}.txt', 'x', encoding="utf-8") as file:
    #     file.write(set_list(my_list))

def calculate_tf_words_each_paragraf(num):
        global list_tf
        list_tf=[]
        with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
         string = tokenizer(doc.read())
         # split each paragraf by \n
         patern_split_paragraf = "[\n]+"
         my_list = re.split(patern_split_paragraf, string)
         patern_split_words = "[,. :;|]+"
         print("\n"*3)
         for i in range(len(my_list)):
           my_list1 = re.split(patern_split_words, my_list[i])
           dict_list1=set_list(my_list1)
           list_tf.append(computeTF(dict_list1,dict_list1))

def open_json():
    with open('data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)


def computeIDF(docList):
    global idfDict
    idfDict = {}
    list_help=[]
    N = len(docList)
    for i in range(N):
     idfDict = dict.fromkeys(docList[i].keys(), 0)
     for word, val in idfDict.items():
        idfDict[word] = log(N/ (float(val) + 1))
        
     list_help.append(idfDict)
     print(len(idfDict))
    return(list_help)




calculate_tf_words_each_paragraf(0)
calculate_idf_words_each_paragraf(0)
# with open(f'data\\document_{0}.txt', "r", encoding="utf-8") as doc:
#         string = tokenizer(doc.read())
#         patern = "[\n|]+"
#         my_list = re.split(patern, string)
#         print(f"\n\n{len(my_list)}")