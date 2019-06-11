import glob
import os
import pickle
import re

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

import numpy as np

from mat2vec.processing.process import MaterialsTextProcessor

text_processing = MaterialsTextProcessor()


COMMON_TERMS = ["-", "-", b"\xe2\x80\x93", b"'s", b"\xe2\x80\x99s", "from", "as", "at", "by", "of", "on",
                "into", "to", "than", "over", "in", "the", "a", "an", "/", "under", ":"]
EXCLUDE_PUNCT = ["=", ".", ",", "(", ")", "<", ">", "\"", "“", "”", "≥", "≤", "<nUm>"]
EXCLUDE_TERMS = ["=", ".", ",", "(", ")", "<", ">", "\"", "“", "”", "≥", "≤", "<nUm>", "been", "be", "are",
                 "which", "were", "where", "have", "important", "has", "can", "or", "we", "our",
                 "article", "paper", "show", "there", "if", "these", "could", "publication",
                 "while", "measured", "measure", "demonstrate", "investigate", "investigated",
                 "demonstrated", "when", "prepare", "prepared", "use", "used", "determine",
                 "determined", "find", "successfully", "newly", "present",
                 "reported", "report", "new", "characterize", "characterized", "experimental",
                 "result", "results", "showed", "shown", "such", "after",
                 "but", "this", "that", "via", "is", "was", "and", "using"]
INCLUDE_PHRASES = ["oxygen_reduction_reaction"]


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def keep_simple_formula(word, count, min_count):
    if text_processing.is_simple_formula(word):
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT


def compute_epoch_accuracies(root, prefix, analogy_file):
    filenames = glob.glob(os.path.join(root, prefix+"_epoch*.model"))
    nr_epochs = len(filenames)
    accuracies = dict()
    losses = [0] * nr_epochs
    for filename in filenames:
        epoch = int(re.search("\d+\.model", filename).group()[:-6])
        m = Word2Vec.load(filename)
        losses[epoch] = m.get_latest_training_loss()
        sections = m.wv.accuracy(analogy_file)
        for sec in sections:
            if sec["section"] not in accuracies:
                accuracies[sec["section"]] = [0] * nr_epochs
            correct, incorrect = len(sec["correct"]), len(sec["incorrect"])
            if incorrect > 0:
                accuracy = correct / (correct + incorrect)
            else:
                accuracy = 0
            accuracies[sec["section"]][epoch] = (correct, incorrect, accuracy)
        save_obj(accuracies, os.path.join("models", prefix + "_accuracies"))
        save_obj(np.concatenate([np.array([losses[0]]), np.diff(losses)]), os.path.join("models", prefix + "_loss"))


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, m):
        output_path = "{}_epoch{}.model".format(self.path_prefix, self.epoch)
        print("Save model to {}.".format(output_path))
        m.save("tmp/"+output_path)
        self.epoch += 1
