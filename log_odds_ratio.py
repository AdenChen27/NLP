# based on https://github.com/kornosk/log-odds-ratio


import math
import numpy as np
import pandas as pd
from collections import Counter

class LogOddsRatio:
    """
    Log-odds-ratio with informative Dirichlet priors
    """

    def __init__(self, corpus_i, corpus_j, background_corpus=None, preprocess_text=False):
        """
        Create a class object and prepare word counts for log-odds-ratio computation

        Args:
            corpus_i:
                A list of documents, each contains a string
            corpus_j:
                A list of documents, each contains a string
            background_corpus (default = None):
                If None, it will be assigned to a concatenation of `corpus_i` and `corpus_j`
        """
        if preprocess_text:
            raise("preprocess text first")

        if background_corpus == None:
            background_corpus = corpus_i + corpus_j

        self.y_i = Counter(" ".join(corpus_i).split())
        self.y_j = Counter(" ".join(corpus_j).split())
        self.alpha = Counter(" ".join(background_corpus).split())


        # Sort dicts
        self.y_i = {k: v for k, v in sorted(self.y_i.items(), key=lambda item: item[1], reverse=True)}
        self.y_j = {k: v for k, v in sorted(self.y_j.items(), key=lambda item: item[1], reverse=True)}
        self.alpha = {k: v for k, v in sorted(self.alpha.items(), key=lambda item: item[1], reverse=True)}

        # Initialize necessary variables
        self.delta = None
        self.sigma_2 = None
        self.z_scores = None

        # Compute
        self._compute_delta()
        self._compute_sigma_2()
        self._compute_z_scores()

        # # filter out words that occurred fewer than 20 times in either corpus. 
        # words_to_keep = [word for word in self.y_i.keys() & self.y_j.keys() if self.y_i[word] >= 20 and self.y_j[word] >= 20]
        # self.z_scores = {word: self.z_scores[word] for word in words_to_keep}

        # Sort dicts
        self.delta = {k: v for k, v in sorted(self.delta.items(), key=lambda item: item[1], reverse=True)}
        self.sigma_2 = {k: v for k, v in sorted(self.sigma_2.items(), key=lambda item: item[1], reverse=True)}
        self.z_scores = {k: v for k, v in sorted(self.z_scores.items(), key=lambda item: item[1], reverse=True)}


        # # Write to files as backup
        # with open("delta.txt", "w") as f:
        #     for k, v in self.delta.items():
        #         f.write(f"{k},{v}\n")
        # with open("sigma_2.txt", "w") as f:
        #     for k, v in self.sigma_2.items():
        #         f.write(f"{k},{v}\n")
        # with open("z_scores.txt", "w") as f:
        #     for k, v in self.z_scores.items():
        #         f.write(f"{k},{v}\n")
        # with open("results/lod-odds-result.csv", "w") as f:
        #     f.write("word,z_score,count1,count2,total_count\n")
        #     for w, z in self.z_scores.items():
        #         f.write(f"{w},{z},{self.y_i.get(w, 0)},{self.y_j.get(w, 0)},{self.alpha.get(w, 0)}\n")


    def _compute_delta(self):
        """ The usage difference for word w among two corpora i and j
        """
        self.delta = dict()
        n_i = sum(self.y_i.values())
        n_j = sum(self.y_j.values())
        alpha_zero = sum(self.alpha.values())
        # print(f"Size of corpus-i: {n_i}")
        # print(f"Size of corpus-j: {n_j}")
        # print(f"Size of background corpus: {alpha_zero}")

        try:
            for w in set(self.y_i) | set(self.y_j): # iterate through all words among two corpora
                first_log = math.log10((self.y_i.get(w, 0) + self.alpha.get(w, 0)) / (n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
                second_log = math.log10((self.y_j.get(w, 0) + self.alpha.get(w, 0)) / (n_j + alpha_zero - self.y_j.get(w, 0) - self.alpha.get(w, 0)))
                self.delta[w] = first_log - second_log
        except ValueError as e:
            print(f"Y-i of the word {w}:", self.y_i.get(w, 0))
            print(f"alpha of the word {w}:", self.alpha.get(w, 0))
            print(f"value:", (self.y_i.get(w, 0) + self.alpha.get(w, 0)) /
                  (n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
            raise e

    def _compute_sigma_2(self):
        """
        Compute estimated values of sigma squared
        """
        self.sigma_2 = dict()
        for w in self.delta:
            self.sigma_2[w] = (1 / (self.y_i.get(w, 0) + self.alpha.get(w, 0))) + (1 / (self.y_j.get(w, 0) + self.alpha.get(w, 0)))

    def _compute_z_scores(self):
        self.z_scores = dict()
        for w in self.delta:
            self.z_scores[w] = self.delta.get(w, 0) / math.sqrt(self.sigma_2.get(w, 0))


def main(corpus_i, corpus_j, background_corpus=None):
    log_odds_ratio = LogOddsRatio(corpus_i, corpus_j, background_corpus)
    data = {}
    # with open("results/log-odds-result.csv", "w") as f:
        # f.write("word,z_score,count1,count2,total_count\n")
            # f.write(f"{w},{z},{log_odds_ratio.y_i.get(w, 0)},{log_odds_ratio.y_j.get(w, 0)},{log_odds_ratio.alpha.get(w, 0)}\n")
    for word, z_score in log_odds_ratio.z_scores.items():
        data[word] = {
            "z_score": z_score, 
            "count1": log_odds_ratio.y_i.get(word, 0), 
            "count2": log_odds_ratio.y_j.get(word, 0), 
            "total_count": log_odds_ratio.alpha.get(word, 0), 
        }
    return pd.DataFrame(data).transpose()


