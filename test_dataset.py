from dataset import *

def test_penntreebank():
    path = "data/penn/valid.txt"
    treebank = PennTreebankDataset(path)

test_penntreebank()
