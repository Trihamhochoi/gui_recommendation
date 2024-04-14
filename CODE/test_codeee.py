import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
# import jieba
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from tqdm.auto import tqdm
import re


def export_dict_gensim(col_values: np.array):
    # Tokenize(split) the sentences into words
    products_gem = [[text for text in x.split()] for x in col_values]
    # remove some special elements in texts
    # number
    products_gem_re = [[re.sub('[0-9]+', '', e) for e in text] for text in products_gem]
    # special symbols
    special_ls = ['', ' ', ',', '.', '...', '-', ':', ';', '?',
                  '%', '_%', '(', ')', '+', '/', 'g', 'ml']
    products_gem_re = [[t.lower() for t in text if t not in special_ls] for text in products_gem_re]

    # Obtain the number of features based on dictionary: Use corpora.Dictionary
    dictionary = corpora.Dictionary(products_gem_re)
    return dictionary
