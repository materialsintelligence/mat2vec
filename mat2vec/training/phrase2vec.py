from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
import gensim
from mat2vec.training.helpers.utils import EpochSaver, compute_epoch_accuracies, \
    keep_simple_formula, load_obj, COMMON_TERMS, EXCLUDE_PUNCT, INCLUDE_PHRASES
import logging
import os
import argparse
import regex
import pickle
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def exclude_words(phrasegrams, words):
    """Given a list of words, excludes those from the keys of the phrase dictionary."""
    new_phrasergrams = {}
    words_re_list = []
    for word in words:
        we = regex.escape(word)
        words_re_list.append("^" + we + "$|^" + we + "_|_" + we + "$|_" + we + "_")
    word_reg = regex.compile(r""+"|".join(words_re_list))
    for gram in tqdm(phrasegrams):
        valid = True
        for sub_gram in gram:
            if word_reg.search(sub_gram.decode("unicode_escape", "ignore")) is not None:
                valid = False
                break
            if not valid:
                continue
        if valid:
            new_phrasergrams[gram] = phrasegrams[gram]
    return new_phrasergrams


# Generating word grams.
def wordgrams(sent, depth, pc, th, ct, et, ip, d=0):
    if depth == 0:
        return sent, None
    else:
        """Builds word grams according to the specification."""
        phrases = Phrases(
            sent,
            common_terms=ct,
            min_count=pc,
            threshold=th)

        grams = Phraser(phrases)
        grams.phrasegrams = exclude_words(grams.phrasegrams, et)
        d += 1
        if d < depth:
            return wordgrams(grams[sent], depth, pc, th, ct, et, ip, d)
        else:
            return grams[sent], grams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", required=True, help="The path to the corpus to train on.")
    parser.add_argument("--model_name", required=True, help="Name for saving the model (in the models folder).")
    parser.add_argument("--epochs", default=30, help="Number of epochs.")
    parser.add_argument("--size", default=200, help="Size of the embedding.")
    parser.add_argument("--window", default=8, help="Context window size.")
    parser.add_argument("--min_count", default=5, help="Minimum number of occurrences for word.")
    parser.add_argument("--workers", default=16, help="Number of workers.")
    parser.add_argument("--alpha", default=0.01, help="Learning rate.")
    parser.add_argument("--batch", default=10000, help="Minibatch size.")
    parser.add_argument("--negative", default=15, help="Number of negative samples.")
    parser.add_argument("--subsample", default=0.0001, help="Subsampling rate.")
    parser.add_argument("--phrase_depth", default=2, help="The number of passes to perform for phrase generation.")
    parser.add_argument("--phrase_count", default=10, help="Minimum number of occurrences for phrase to be considered.")
    parser.add_argument("--phrase_threshold", default=15.0, help="Phrase importance threshold.")
    parser.add_argument("-include_extra_phrases",
                        action="store_true",
                        help="If true, will look for all_ents.p and add extra phrases.")
    parser.add_argument("-sg", action="store_true", help="If set, will train a skip-gram, otherwise a CBOW.")
    parser.add_argument("-hs", action="store_true", help="If set, hierarchical softmax will be used.")
    parser.add_argument("-keep_formula", action="store_true",
                        help="If set, keeps simple chemical formula independent on count.")
    parser.add_argument("-notmp", action="store_true", help="If set, will not store the progress in tmp folder.")
    args = parser.parse_args()

    all_formula = []
    if args.keep_formula:
        try:
            all_formula = load_obj(args.corpus + "_formula")  # list of formula is supplied

            def keep_formula_list(word, count, min_count):
                if word in all_formula:
                    return gensim.utils.RULE_KEEP
                else:
                    return gensim.utils.RULE_DEFAULT
            trim_rule_formula = keep_formula_list
            logging.info("Using a supplied list of formula to keep simple formula.")
        except:
            # no list is supplied, use the simple formula rule
            trim_rule_formula = keep_simple_formula
            logging.info("Using a function to keep material mentions.")
    else:
        logging.info("Basic min_count trim rule for formula.")
        trim_rule_formula = None

    # The trim rule for extra phrases to always keep them, similar to the formulae.
    if args.include_extra_phrases:
        INCLUDE_PHRASES_SET = set(INCLUDE_PHRASES)
        try:
            with open("all_ents.p", "rb") as f:
                INCLUDE_PHRASES += list(set(pickle.load(f)))
                INCLUDE_PHRASES_SET = set([ip.replace("_", "$@$@$") for ip in INCLUDE_PHRASES])
                logging.info("Included the supplied {} additional phrases.".format(len(INCLUDE_PHRASES)))
        except:
            logging.info("No specific phrases supplied, using the defaults.")

        def keep_extra_phrases(word, count, min_count):
            if word in INCLUDE_PHRASES_SET or trim_rule_formula is not None and \
                    trim_rule_formula(word, 1, 2) == gensim.utils.RULE_KEEP:
                return gensim.utils.RULE_KEEP
            else:
                return gensim.utils.RULE_DEFAULT

        trim_rule = keep_extra_phrases
        logging.info("Keeping the extra phrases independent on their count.")
    else:
        trim_rule = trim_rule_formula
        logging.info("Not including extra phrases, option not specified.")

    # Excluding all formula from the phrases.
    formula_counts = [0] * len(all_formula)
    for i, formula in enumerate(all_formula):
        for writing in all_formula[formula]:
            formula_counts[i] += all_formula[formula][writing]
    formula_strings = [formula for i, formula in enumerate(all_formula) if formula_counts[i] > int(args.phrase_count)]

    # Loading text and generating the phrases.
    sentences = LineSentence(args.corpus)

    # Pre-process everything to force the supplied phrases before it even goes to the phraser.
    processed_sentences = sentences
    if args.include_extra_phrases:
        phrases_by_length = dict()
        for phrase in INCLUDE_PHRASES:
            phrase_split = phrase.split("_")
            if len(phrase_split) not in phrases_by_length:
                phrases_by_length[len(phrase_split)] = [phrase]
            else:
                phrases_by_length[len(phrase_split)].append(phrase)
        max_len = max(phrases_by_length.keys())

        processed_sentences = []
        for sentence in tqdm(sentences):
            for cl in reversed(range(2, max_len + 1)):
                repl_phrases = set(phrases_by_length[cl])
                si = 0
                while si <= len(sentence) - cl:
                    if "_".join(sentence[si:cl + si]) in repl_phrases:
                        sentence[si] = "$@$@$".join(sentence[si:cl + si])
                        del(sentence[si + 1:cl + si])
                    else:
                        si += 1
            processed_sentences.append(sentence)

    # Process sentences to force the extra phrases.
    sentences, phraser = wordgrams(processed_sentences,
                          depth=int(args.phrase_depth),
                          pc=int(args.phrase_count),
                          th=float(args.phrase_threshold),
                          ct=COMMON_TERMS,
                          et=EXCLUDE_PUNCT + formula_strings,
                          ip=INCLUDE_PHRASES)
    phraser.save(os.path.join("models", args.model_name + "_phraser.pkl"))

    if not args.notmp:
        callbacks = [EpochSaver(path_prefix=args.model_name)]
    else:
        callbacks = []
    my_model = Word2Vec(
        sentences,
        size=int(args.size),
        window=int(args.window),
        min_count=int(args.min_count),
        sg=bool(args.sg),
        hs=bool(args.hs),
        trim_rule=trim_rule,
        workers=int(args.workers),
        alpha=float(args.alpha),
        sample=float(args.subsample),
        negative=int(args.negative),
        compute_loss=True,
        sorted_vocab=True,
        batch_words=int(args.batch),
        iter=int(args.epochs),
        callbacks=callbacks)
    my_model.save(os.path.join("models", args.model_name))

    analogy_file = os.path.join("data", "analogies.txt")
    # Save the accuracies in the tmp folder.
    compute_epoch_accuracies("tmp", args.model_name, analogy_file)
