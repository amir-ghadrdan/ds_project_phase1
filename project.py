import re
import json
from math import log
from collections import Counter



def tokenizer(text):
    return text.lower()


def set_list(my_list):

    return dict(Counter(my_list))


def open_doc_read(num):
    with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
        string = tokenizer(doc.read())
        patern = "[,. :;\n|]+"
        my_list = re.split(patern, string)
    print(set_list(my_list))
    print(len(set_list(my_list).keys()))
    # with open(f'file_data\\file_{num}.txt', 'x', encoding="utf-8") as file:
    #     file.write(set_list(my_list))


def open_json():
    with open('data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)


def tf_idf():
    pass


open_doc_read(0)
