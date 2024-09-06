from gensim.models import phrases, word2vec
import nltk
import numpy as np
import codecs
from collections import Counter
from nltk.corpus import stopwords
import os
import re
import string

punct_chars = list((set(string.punctuation) | {'»', '–', '—', '-',"­", '\xad', '-', '◾', '®', '©','✓','▲', '◄','▼','►', '~', '|', '“', '”', '…', "'", "`", '_', '•', '*', '■'} - {"'"}))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
printable = set(string.printable)

stopwords = set(stopwords.words('english'))


def clean_text(text, remove_numeric=True):
    # lower case
    text = text.lower()
    
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    
    # substitute all other punctuation with whitespace
    text = replace.sub(' ', text)
    
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # strip off spaces on either end
    text = text.strip()
    
    # make sure all chars are printable
    text = ''.join([c for c in text if c in printable])
    words = text.split()
    
    if remove_numeric:
        words = [w for w in words if not w.isdigit()]
    
    return words


def main(texts, output_dir="data/wv", bootstrap=True, num_runs=50, dim=100, window=5):
    """Runs word2vec training on data.

    Args:
        texts: list of texts (should be cleaned)
        bootstrap: whether to bootstrap sample from the sentences

    """
    os.makedirs(output_dir, exist_ok=True)

    all_sentences = []
    for text in texts:
        all_sentences.extend([clean_text(s) for s in nltk.sent_tokenize(text)])
    # print(all_sentences)

    # Create model
    bigrams = phrases.Phrases(all_sentences, min_count=5, delimiter=' ')
    # , common_terms=stopwords)

    # Create vocabulary of bigrams
    print("Creating vocabulary...")
    vocab = [w for sent in bigrams[all_sentences] for w in sent]
    vocab = [w for w, count in Counter(vocab).most_common() if count >= 5]

    # Save vocab
    with codecs.open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    # Run word2vec model
    all_sentences = np.asarray(all_sentences, dtype="object")
    for run_idx in range(num_runs):
        print("Run #%d" % run_idx)
        if bootstrap:
            # print(np.random.choice(np.asarray(all_sentences, dtype="object"), len(all_sentences), replace=True))

            # data = bigrams[np.random.choice(all_sentences, len(all_sentences), replace=True)]
            data = bigrams[np.random.choice(all_sentences, len(all_sentences), replace=True)]
        else:
            data = bigrams[all_sentences]
        model = word2vec.Word2Vec(data, vector_size=dim, window=window, sg=1, min_count=5, workers=10)
        model.wv.save(os.path.join(output_dir, str(run_idx) + '.wv'))




