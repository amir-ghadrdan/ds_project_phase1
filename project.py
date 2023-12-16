import re
import json


def tokenizer(text):
    return text.lower()


def set_list(my_list):
    my_dict = {}
    for item in my_list:
        my_dict[tokenizer(item)] = True
    unique_list = list(my_dict.keys())
    return (str(unique_list))


def open_doc_read(num):
    with open(f'data\\document_{num}.txt', "r", encoding="utf-8") as doc:
        string = doc.read()
        patern = "[,. :;\n|]+"
        my_list = re.split(patern, string)

    with open(f'file_data\\file_{num}.txt', 'x', encoding="utf-8") as file:
        file.write(set_list(my_list))


def open_json():
    with open('data.json', "r+", encoding='utf-8') as data:
        new_data = json.load(data)


def tf_idf():
    pass


open_json()
