import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def tokenizer(text):
    translation_table = str.maketrans("", "", string.punctuation)
    text_without_punctuation = text.translate(translation_table)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text_without_punctuation)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words
