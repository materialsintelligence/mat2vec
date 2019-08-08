"""Computes metrics of the word2vec using PMI relations as references."""
import argparse
import collections
import datetime
import json
import logging
import multiprocessing
import os
import time

from functools import partial
from gensim.models import Word2Vec
from tqdm import tqdm
from typing import Dict
from typing import List
from typing import Set


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)


def find_entity(entities: str, vocab: Set[str]) -> str:
    """Returns the first match of a list of entities in a vocabulary.

    Args:
        entities: A list of entities.
        vocab: A set of ngrams.

    Returns:
        entity: The first match of `entities` in `vocab`. Returns `None`
            if there was no match.
    """
    for entity in entities.split('|'):
        entity = entity.strip()
        if entity in vocab:
            return entity
    return None


def load_entities_fn(file_name: str, types: List[str]) -> Dict[str, Set[str]]:
    """Given a json-lines file from Pubtator, loads the entities.

    Args:
        file_name: Path to the json-lines files from Pubtator.
        types: List of entity types to be loaded.

    Returns:
        entities: A dictionary where key is an entity type and value is a
            set of entities of that type.
    """
    entities = collections.defaultdict(set)
    with open(file_name) as f:
        for line in f:
            json_obj = json.loads(line)
            for entity in json_obj['entity_list']:
                if entity['type'].lower() in types:
                    for entity_i in entity['name'].lower().split('|'):
                        entities[entity['type'].lower()].add(entity_i.strip())
    return entities


def load_entities(folder: str, types: List[str]) -> List[Set[str]]:
    """Given a folder with json-lines files from Pubtator, loads the entities.

    Args:
        folder: Path a folder containg json-lines files from Pubtator.
        types: List of entity types to be loaded.

    Returns:
        entities: A list of sets whose entities follow the same order from
            `types`.
    """
    file_names = [
        os.path.join(folder, file_name) for file_name in os.listdir(folder)
        if file_name.endswith('.json')
    ]

    pool = multiprocessing.Pool()
    load_entities_fn_partial = partial(load_entities_fn, types=types)

    results = pool.map(load_entities_fn_partial, file_names)
    pool.close()
    pool.join()

    # Merge sets.
    entities = {type_: set() for type_ in types}
    entities = collections.defaultdict(set)
    for partial_result in results:
        for type_ in types:
            entities[type_].update(partial_result[type_])

    # Convert to list.
    return [entities[type_] for type_ in types]


def main(model_path: str, ref_pairs_path: str, bioconcepts_path: str,
         k: int = 100, output_path=None) -> None:

    if output_path:
        output_file = open(output_path, 'w')

    w2v_model = Word2Vec.load(model_path)
    entities1_found = 0
    entities2_found = 0
    unique_entities1 = set()
    unique_entities2 = set()
    pairs_found = 0
    pairs_total = 0
    recall = 0
    MRR = 0

    start_time = time.time()
    diseases = load_entities(folder=bioconcepts_path, types=['disease'])[0]
    logging.info('Loaded entities {} in {}'.format(
        len(diseases), datetime.timedelta(seconds=time.time() - start_time)))

    with open(ref_pairs_path) as f:
        for line in tqdm(f):
            line = line.lower()
            line = line.replace('(gene)', '')
            line = line.replace('(disease)', '')
            line = ' '.join(line.split())
            line = ' '.join(line.split(':')[:-1])  # Remove PMI score.
            entities1, entities2 = line.strip().split(' [sep] ')

            unique_entities1.add(entities1)
            unique_entities2.add(entities2)

            canonical_entity1 = find_entity(entities1, w2v_model.wv.vocab)
            canonical_entity2 = find_entity(entities2, w2v_model.wv.vocab)

            pairs_total += 1
            if canonical_entity1:
                entities1_found += 1
            if canonical_entity2:
                entities2_found += 1
            if canonical_entity1 and canonical_entity2:
                pairs_found += 1
                word2vec_entity1 = canonical_entity1.replace(' ', '_')
                rank = 0
                for candidate_entity2, score in w2v_model.wv.most_similar(
                        word2vec_entity1, topn=1000 * k):
                    candidate_entity2 = candidate_entity2.replace('_', ' ')
                    if candidate_entity2 in diseases:
                        rank += 1
                        label = False
                        if candidate_entity2 == canonical_entity2:
                            recall += 1
                            MRR += 1 / rank
                            label = True

                        if output_file:
                            output_file.write('{}\t{}\t{}\t{}\n'.format(
                                canonical_entity1,
                                candidate_entity2,
                                rank,
                                label))
                        if rank >= k:
                            break

        MRR = MRR / pairs_found
        recall = recall / pairs_found

        logging.infor(
            f'Entity 1 found/total: {entities1_found}/{len(unique_entities1)}')
        logging.info(
            f'Entity 2 found/total: {entities2_found}/{len(unique_entities2)}')
        logging.info(f'Pairs found/total: {pairs_found}/{pairs_total}')
        logging.info(f'Recall: {recall}')
        logging.info('MRR: {MRR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add description here.')

    parser.add_argument('--model', required=True,
                        help='Path to the pretrained word2vec model.')
    parser.add_argument('--ref_pairs', required=True,
                        help='Reference file containing the entity pairs '
                             'separated by [SEP].')
    parser.add_argument(
        '--bioconcepts_path',
        required=True,
        help='BioConcepts file containing the entity and types.')
    parser.add_argument('--output',
                        help='Output file to write the top retrived words '
                             'from word2vec. '
                             'Do do write if not passed')
    parser.add_argument('--k', default=100, type=int,
                        help='Top-K word2vec similarities.')

    args = parser.parse_args()

    logging.info(args)
    main(model_path=args.model,
         ref_pairs_path=args.ref_pairs,
         bioconcepts_path=args.bioconcepts_path,
         k=args.k,
         output_path=args.output)
