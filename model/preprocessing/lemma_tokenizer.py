
import string

import nltk
from nltk import pos_tag, word_tokenize, WordNetLemmatizer, sent_tokenize
from nltk.corpus import wordnet

from model.preprocessing.base_tokenizer import Tokenizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


class LemmaTokenizer(Tokenizer):

    def __init__(self, features=None):
        self.features = features
        with open('data/stopwords.txt', 'r') as f:
            self.stopwords = f.read().splitlines()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, documents: list):
        tokenized_docs = []
        doc_nbr = 0
        print(f' Tokenizing : {len(documents)} documents')
        for doc in documents:
            doc_nbr += 1
            if doc_nbr % 1000 == 0:
                print(f'tokenized {round((doc_nbr / len(documents)) * 100, 2)} % ')
            sentence_tokens = [word_tokenize(t) for t in sent_tokenize(doc)]
            pos_sentence_tokens = [pos_tag(tokens) for tokens in sentence_tokens]
            pos_tokens = [post_token for sentence_tokens in pos_sentence_tokens for post_token in sentence_tokens]
            exclude = set(string.punctuation)
            pos_tokens = [pos_token for pos_token in pos_tokens
                          if not pos_token[0].lower() in self.stopwords and
                          pos_token[0] not in exclude and
                          pos_token[0] is not string.digits]
            doc_tokens = [self.lemmatizer.lemmatize(post_token[0], self.get_wordnet_pos(post_token[1])) for post_token
                          in pos_tokens]
            doc_tokens = [token for token in doc_tokens if token.isalpha()]
            doc_tokens = [token for token in doc_tokens if token not in self.stopwords]
            if len(doc_tokens) == 0:
                continue
            tokenized_docs.append(doc_tokens)

        return tokenized_docs

    def get_wordnet_pos(self, tag: string):

        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
