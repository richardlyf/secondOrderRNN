import torchtext
EMBEDDING = 'glove.840B.300d'

def build_dataset(args, is_parens=False):
	"""
	Constructs a torchtext Field and train, validation, and test datasets
	@param args All command line arguments
	@param is_parens Boolean, indicates whether to process parentheses dataset

	@return TEXT torchtext Field object
	@return (train, val, test) torchtext datasets for training, validation, testing
	"""
    TEXT = torchtext.data.Field()
    if is_parens: 
    	TEXT.tokenize = tokenize_parens

    # construct Dataset objects, with TEXT as the datatype
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=".", 
        train=args.train_path, 
        validation=args.valid_path, 
        test=args.valid_path, 
        text_field=TEXT)

    # map vocabulary words to indices
    TEXT.build_vocab(train)
    
    # use glove embedding for word vectors 
    if not is_parens:
    	TEXT.vocab.load_vectors(EMBEDDING)

    return TEXT, (train, val, test)

def tokenize_parens(string):
	"""
	Tokenizer function for synthetic parenthesis dataset
	"""
	return string.replace("END", "").split()