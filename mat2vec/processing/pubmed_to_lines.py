"""Converts PubMed abstracts to a file with one abstract per line."""
import argparse
import datetime
import gzip
import logging
import multiprocessing
import os
import process
import time
import xml.etree.cElementTree as ET

from functools import partial
from typing import Tuple


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)

# Approximate total number of abstracts so we can estimate processing time
# without having to read all files beforehand.
TOTAL_DOCS = 32000000


def abstracts_to_lines_fn(
        args: Tuple[str],
        text_processor: process.MaterialsTextProcessor) -> int:
    """Write abstracts in a PubMed XML file to a line-separated text file.

    Args:
        args: A tuple containing the input and output file paths.
        text_processor: an instance of process.MaterialsTextProcessor.

    Returns:
        doc_count: number of abstracts written to the file.
    """
    input_file, output_file = args
    fout = open(output_file, 'w')
    doc_count = 0
    with gzip.open(input_file, 'rt') as f:
        text = f.read()
        tree = ET.ElementTree(ET.fromstring(text))
        root = tree.getroot()

        for abstract_text in root.iterfind('.//AbstractText'):
            doc_text = abstract_text.text
            if doc_text is None:
                continue
            doc_tokens, _ = text_processor.process(doc_text)
            doc_text = ' '.join(doc_tokens)

            fout.write('{}\n'.format(doc_text))
            doc_count += 1
            # if doc_count >= 1000:
            #    break

    fout.close()
    return doc_count


def abstracts_to_lines(input_folder: str, output_folder: str,
                       n_processes: int) -> None:
    """Write abstracts in PubMed XML files to line-separated text files.

    Args:
        input_folder: Path to the folder containing the PubMed XML files.
        output_folder: Path to the folder to write the raw text files.
        n_processes: Number of processes to read and write the files in
            parallel.
    """

    start_time = time.time()
    input_files = []
    output_files = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.gz'):
            input_files.append(os.path.join(input_folder, file_name))
            output_files.append(os.path.join(
                output_folder, file_name.replace('.xml.gz', '.txt')))

    logging.info('{} files found'.format(len(input_files)))
    text_processor = process.MaterialsTextProcessor()
    abstracts_to_lines_fn_partial = partial(
        abstracts_to_lines_fn,
        text_processor=text_processor)

    # Warning: more than 8 processes causes OOM on a 64GB machine.
    pool = multiprocessing.Pool(n_processes)
    doc_count = 0
    for partial_doc_count in pool.imap_unordered(
            abstracts_to_lines_fn_partial, zip(input_files, output_files)):
        doc_count += partial_doc_count
        time_passed = time.time() - start_time
        time_remaining = TOTAL_DOCS * time_passed / doc_count - time_passed
        logging.info('Processed {}/{} docs in {} (Remaining: {})'.format(
            doc_count, TOTAL_DOCS,
            datetime.timedelta(seconds=time_passed),
            datetime.timedelta(seconds=time_remaining)))

    pool.close()
    pool.join()

    logging.info('Done. Time: {}'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    logging.info('Converted {} docs in {}'.format(
        doc_count,
        datetime.timedelta(seconds=time.time() - start_time)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Converts PubMed abstracts to a file with one abstract '
                    'per line.')
    parser.add_argument('--input_folder', required=True,
                        help='Folder containing the XML.gz files.')
    parser.add_argument('--output_folder', required=True,
                        help='output folder.')
    parser.add_argument(
        '--n_processes', default=None, type=int,
        help='Number of processes to read and write files in parallel. '
             'Warning: more than 8 processes causes OOM on a 64GB machine.')
    args = parser.parse_args()

    logging.info(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    abstracts_to_lines(input_folder=args.input_folder,
                       output_folder=args.output_folder,
                       n_processes=args.n_processes)
    logging.info('Done!')
