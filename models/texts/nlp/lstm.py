import torch
import torch.nn as nn
import gensim
import numpy as np
from .bert_embeddings import bert_embeddings
import pdb
torch.backends.cudnn.enabled = False


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.w2v = Word2Vec()
        # self.ft = FasttextEmbeddings()

        self.input_size = 300
        self.dec = nn.LSTM(input_size=self.input_size,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True).double()

    def sort(self, x, reverse=False):
        return zip(*sorted([(x[i], i) for i in range(len(x))], reverse=reverse))

    def sortNpermute(self, x, mask):
        mask_sorted, perm = self.sort(
            mask.sum(dim=-1).cpu().numpy(), reverse=True)
        return x[list(perm)], list(mask_sorted), list(perm)

    def inverse_sortNpermute(self, x, perm):
        _, iperm = self.sort(perm, reverse=False)
        if isinstance(x, list):
            return [x_[list(iperm)] for x_ in x]
        else:
            return x[list(iperm)]

    def forward(self, sentences):
        x_orig, mask_orig = self.w2v(sentences)
        x, mask, perm = self.sortNpermute(x_orig, mask_orig)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True)

        ''' forward pass through lstm '''
        x, (h, m) = self.dec(x)

        ''' get the output at time_step=t '''
        h = self.inverse_sortNpermute(h[-1], perm)

        return h, x_orig


class BERTSentenceEncoderLSTM(nn.Module):
    def __init__(self, hidden_size, lstm_layers=2):
        super(BERTSentenceEncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bert = bert_embeddings()

        self.input_size = 4096
        self.dec = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)

    def get_output_dim(self):
        return self.hidden_size

    def sort(self, x, reverse=False):
        return zip(*sorted([(x[i], i) for i in range(len(x))], reverse=reverse))

    def sortNpermute(self, x, mask):
        mask_sorted, perm = self.sort(
            mask.sum(dim=-1).cpu().numpy(), reverse=True)
        return x[list(perm)], list(mask_sorted), list(perm)

    def inverse_sortNpermute(self, x, perm):
        _, iperm = self.sort(perm, reverse=False)
        if isinstance(x, list):
            return [x_[list(iperm)] for x_ in x]
        else:
            return x[list(iperm)]

    def forward(self, sentences):
        sentences = [x_.split(' ') for x_ in sentences]
        x_orig, mask_orig = self.bert.get_vectors(sentences)
        # x_ = self.lin(x_orig)
        # print(x_.shape)
        ''' Here you will find why sort and permute is done
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence'''
        x, mask, perm = self.sortNpermute(x_orig, mask_orig)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True)

        ''' forward pass through lstm '''
        x, (h, m) = self.dec(x)

        ''' get the output at time_step=t '''
        h = self.inverse_sortNpermute(h[-1], perm)

        # print(h1.shape)

        return h


class BaseTokenizer():
    def __init__(self, vocab):
        self.vocab = vocab
        self.hidden_size = 300
        self._UNK = '_UNK'
        self._SEP = '_SEP'
        self.random_vec = torch.rand(self.hidden_size)
        self.zero_vec = torch.zeros(self.hidden_size)

    def tokenize(self, sentence):
        words_ = sentence.split(' ')

        ''' Lowercase all words '''
        words_ = [w.lower() for w in words_]

        ''' Add _UNK for unknown words '''
        words = []
        for word in words_:
            if word in self.vocab:
                words.append(word)
            else:
                words.append('_UNK')
        return words


class Word2Vec(nn.Module):
    '''
    Take a bunch of sentences and convert it to a format that Bert can process
    * Tokenize
    * Add _UNK for words that do not exist
    * Create a mask which denotes the batches
    '''

    def __init__(self, path2file='nlp/GoogleNews-vectors-negative300.bin.gz'):
        super(Word2Vec, self).__init__()
        self.dummy_param = nn.Parameter(torch.Tensor([1]))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            path2file, binary=True)
        print('Loaded Word2Vec model')

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BaseTokenizer(self.model.vocab)

    def __call__(self, x):
        self.device = self.dummy_param.device
        x = [self.tokenizer.tokenize(x_) for x_ in x]
        max_len = max([len(x_) for x_ in x])

        mask = torch.Tensor([[1]*len(x_) + [0]*(max_len-len(x_))
                             for x_ in x]).long().to(self.device)
        x = [x_ + ['_SEP']*(max_len-len(x_)) for x_ in x]
        vectors = []
        for sentence in x:
            vector = []
            for word in sentence:
                if word == self.tokenizer._UNK:
                    vector.append(self.tokenizer.random_vec)
                elif word == self.tokenizer._SEP:
                    vector.append(self.tokenizer.zero_vec)
                else:
                    vector.append(torch.from_numpy(self.model.word_vec(word)))
            vector = torch.stack(vector, dim=0).double().to(self.device)
            vectors.append(vector)
        vectors = torch.stack(vectors, dim=0).double()
        return vectors, mask


if __name__ == "__main__":

    to_get_embeddings = Word2Vec()

    # use batch_to_ids to convert sentences to character ids
    sentences = ['A person is walking forwards and waving his hand',
                 'A human is walking in a circle', 'A person is playing violin while singing']

    # print(character_ids)

    embeddings, mask = to_get_embeddings(sentences)

    # print(embeddings)
    print(embeddings.size())
