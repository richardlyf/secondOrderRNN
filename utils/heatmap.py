# -*- coding: utf-8 -*-
# Source: https://github.com/jiesutd/Text-Attention-Heatmap-Visualization/blob/master/text_attention.py
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12

import argparse
import numpy as np
import os

def argParser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", dest="infile", type=str, default="", help="Path to the text file containing x and alpha")
    parser.add_argument("--output", dest="outpath", type=str, default="./data/heatmaps", help="Path to save latex files.")
    parser.add_argument("--prefix", dest="prefix", type=str, default="sample", help="Prefix for LaTeX file names.")
    args = parser.parse_args()
    return args


def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list


def generate_latex(text_list, attention_list, color_list, latex_file, rescale_value = False):
    """
    Convert the text/attention list to latex code, which will further generates the text 
    heatmap based on attention weights.
    @param text_list (list[str]):
    @param attention_list (list[float]):
    @param color_list (list[str]):
    @param latex_file (str):
    @param rescale_value (bool):
    """
    # confirm that lists are the same sizes
    assert(len(text_list) == len(attention_list))
    assert(len(color_list) == len(attention_list))

    # rescale attention values
    if rescale_value:
        attention_list = rescale(attention_list)

    # escape latex special characters in the text
    text_list = clean_word(text_list)

    word_num = len(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{adjustbox}
\definecolor{color1}{RGB}{77,175,74} % green
\definecolor{color2}{RGB}{255,255,179} % yellow
\definecolor{color3}{RGB}{152,78,163} % purple
\definecolor{color4}{RGB}{251,128,114} % red
\definecolor{color5}{RGB}{128,177,211} % blue
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            string += "\\colorbox{%s!%s}{"%(color_list[idx], attention_list[idx])+"\\strut " + text_list[idx]+"} "
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''
\end{document}''')


def sentence_to_list(line):
    """
    Clean one sentence from the batch
    """
    clean = line[1:].strip() \
                    .replace("]", "") \
                    .replace("[", "") \
                    .split(", ")
    return [c[1:-1] for c in clean]


def attention_to_numpy(att):
    """
    Convert one (batch_size x number of cells) attention array to numpy
    """
    lines = att.strip().replace("[[", "").replace("]]","").split("], [")
    array = np.stack([np.fromstring(l, sep = ",") for l in lines])
    return array

def get_sentences(file):
    """
    Read first batch of sentences from text file
    """
    sentences = []
    attention = []
    with open(file, 'r') as f:
        # skip first few printed lines
        line = f.readline()
        while line[0] != '0':
            line = f.readline()
        # read in first batch of sentences
        while line[0] != '[':
            sentences.append(sentence_to_list(line))
            line = f.readline()
        # read in num_words number of attention arrays
        num_words = len(sentences[0])    
        print(num_words)
        for i in range(num_words):
            attention.append(attention_to_numpy(line))
            line = f.readline()
    attention = np.stack(attention)
    return sentences, attention


def preprocess(sentences, attention, idx):
    """
    Generate text_list, attention_list, and color_list for generate_latex
    """
    colors = ['color1', 'color2', 'color3', 'color4', 'color5']
    text_list = sentences[idx]
    seq_len = len(text_list)
    att = attention[:, idx, :]
    color_list = []
    attention_list = []
    for word_idx in range(seq_len):
        attention_list.append(max(att[word_idx, :])*100)
        color_list.append(colors[np.argmax(att[word_idx, :])])
    return text_list, attention_list, color_list


def main():
    args = argParser()
    sentences, attention = get_sentences(args.infile)
    for idx in range(len(sentences)):
        text_list, attention_list, color_list = preprocess(sentences, attention, idx)
        path = os.path.join(args.outpath, "{}_sent{}.tex".format(args.prefix, idx))
        generate_latex(text_list, attention_list, color_list, latex_file=path)
    # convert latex to PDF
    os.system('cd {}; for i in *.tex; do pdflatex $i;done'.format(args.outpath))
    # remove unneeded files
    os.system('cd {}; rm *.tex; rm *.aux; rm *.log'.format(args.outpath))

if __name__ == '__main__':
    main()