# Targeted Syntactic Evaluation of LMs
This directory contains data and evaluation code for the following paper:

R. Marvin and T. Linzen. 2018. Targeted Syntactic Evaluation of Language Models. Proceedings of EMNLP. 

The content of this directory is adopted from repository:

[LM_syneval](https://github.com/richardlyf/LM_syneval)


## HOW TO USE THIS CODE

1. Makes templates.

You can make the default templates by running:
```
python make_templates.py $TEMPLATE_DIR
```

This generates `.pickle` files which contants the templates.
Currently there should be templates already generated in `EMNLP2018/templates`

2. Generate test sentences from the templates

Create `all_test_sents.txt` which contains all test sentences:
```
python LM_eval.py --generate_test_sents
```

3. Test the model on `all_test_sents.txt`

Run `run.py` in test mode and set `all_test_sents.txt` as the test path. Specify `--stats-output-file rnn.output` so the file will contain the stats of the model 

4. Score the generated stats

Generate scores for the model statistics and get output `RNN_results.pickle`:
```
python LM_eval.py --input_file rnn.output
```

5. Analyze the results

You can run
```
python analyze_results.py --results_file $RESULTS_FILE --model_type $TYPE --mode $MODE
```
where $RESULTS is the path to the file where the RNN LM/multitask or ngram results were dumped, $MODEL_TYPE is the type of model (rnn/multitask/ngram) and $MODE is an optional argument that can be 'full' or 'condensed' or 'overall' (default is 'overall').

At the 'condensed' level, we report the accuracies for each sub-division of the sentences we're interested in analyzing. For example, you might want to know the percent of time the model preferred the grammatical over ungrammatical sentence when there was a main subject/verb mismatch vs. the percent of time the model preferred the grammatical over ungrammatical sentence when there was an embedded subject/verb mismatch. This info level suppresses individual sentence pairs and shows you the scores for these subgroups.

In addition to the scores above, at the 'full' level, we will sample some sentences that the model gets correct/incorrect and display the log probabilities of the model at each word in the sentence. This kind of measure can be used to determine if there are strong lexical biases in the model that have an effect on the overall performance.

If you don't specify a 'full' or 'condensed' info level, then only the total accuracies for each test will be reported.

The full list of commands for analyzing results can be found by typing
```
python analyze_results.py -h
```
For example, if you specify --anim, you can look at whether the animacy of the subjects/verbs has an effect on the language models' ability to succeed at certain syntactic constructions. 