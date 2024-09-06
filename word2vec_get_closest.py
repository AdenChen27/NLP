import argparse
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

def get_models(filelist):
    model_files = [f for f in filelist if f.endswith('.wv')]
    models = [KeyedVectors.load(fname, mmap='r') for fname in model_files]
    return models

def get_closest(queries, models, vocab, idx2word, n=15):
    cosines = []
    for m in models:
        cosines.append([np.mean([m.similarity(q, word) for q in queries]) for word in vocab])
    cosines = np.mean(np.array(cosines), axis=0)
    # return [(idx2word[idx], cosines[idx]) for idx in cosines.argsort()[-20:][::-1]]
    return [(idx2word[idx], cosines[idx]) for idx in cosines.argsort()[-n:][::-1]]

def main(queries, word2vec_dir="data/wv", n=15):
    # print("Loading models...")
    filelist = []
    for subdir, dirs, files in os.walk(word2vec_dir):
        for file in files:
            filelist.append(os.path.join(subdir, file))
    models = get_models(filelist)

    # Get vocab
    vocab = set(models[0].key_to_index)
    for m in models:
        vocab &= set(m.key_to_index)

    # Remove queries not in vocab
    queries = set(queries)
    not_in_vocab = queries - vocab
    if not_in_vocab:
        print("Not in vocab:", not_in_vocab)
    queries = list(queries - not_in_vocab)
    vocab = list(vocab)
    idx2word = {i: w for i, w in enumerate(vocab)}

    # print("Getting most similar words...")
    closest = get_closest(queries, models, vocab, idx2word, n)

    ret_w = []
    ret_c = []
    for (w, c) in closest:
        # print("%s %.2f" % (w, c))
        ret_w.append(w)
        ret_c.append(c)

    ret = pd.DataFrame({"word": ret_w, "d": ret_c}).set_index("word")
    ret.index.name = None
    return ret


