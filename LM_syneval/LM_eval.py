import argparse
import pickle
import os
import subprocess
import operator
import logging
from progress.bar import Bar
from tester.TestWriter import TestWriter
from template.TestCases import TestCase

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Parameters for testing a language model")

parser.add_argument('--template_dir', type=str, default='EMNLP2018/templates',
                    help='Location of the template files')
parser.add_argument('--test_sents_file', type=str, default='all_test_sents.txt',
                    help='File with all of the sentences that will be tested')
parser.add_argument('--input_file', type=str, default='rnn.output',
                    help='File that stores all the statistics of the model')

args = parser.parse_args()

writer = TestWriter(args.template_dir, args.test_sents_file)
testcase = TestCase()
tests = testcase.all_cases

all_test_sents = {}
for test_name in tests:
    test_sents = pickle.load(open(args.template_dir+"/"+test_name+".pickle", 'rb'))
    all_test_sents[test_name] = test_sents

unit_type = "word"
writer.write_tests(all_test_sents, unit_type)
name_lengths = writer.name_lengths
key_lengths = writer.key_lengths


def main():
    logging.info("Testing RNN...")
    results = score_rnn()
    with open("RNN_results.pickle", 'wb') as f:
        pickle.dump(results, f)


def score_rnn():
    logging.info("Scoring RNN...")
    with open(args.input_file, 'r') as f:
        all_scores = {}
        first = False
        score = 0.
        sent = []
        prev_sentid = -1
        for line in f:
            if line.strip() == "":
                first = True
            elif "===========================" in line:
                first = False
                break
            elif first and len(line.strip().split()) == 6 and "torch.cuda" not in line:
                wrd, sentid, wrd_score = [line.strip().split()[i] for i in [0,1,4]]
                score = -1 * float(wrd_score) # multiply by -1 to turn surps back into logprobs
                sent.append((wrd, score))
                if wrd == ".":
                    name_found = False
                    for (k1,v1) in sorted(name_lengths.items(), key=operator.itemgetter(1)):
                        if float(sentid) < v1 and not name_found:
                            name_found = True
                            if k1 not in all_scores:
                                all_scores[k1] = {}
                            key_found = False
                            for (k2,v2) in sorted(key_lengths[k1].items(), key=operator.itemgetter(1)):
                                if int(sentid) <  v2 and not key_found:
                                    key_found = True
                                    if k2 not in all_scores[k1]:
                                        all_scores[k1][k2] = []
                                    all_scores[k1][k2].append(sent)
                    sent = []
                    if float(sentid) != prev_sentid+1:
                        logging.info("Error at sents " + str(sentid) + " and " + str(prev_sentid))
                    prev_sentid = float(sentid)
    return all_scores


if __name__ == '__main__':
    main()