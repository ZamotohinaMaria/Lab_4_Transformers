import torch, string, gc, tqdm, math, os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple, List


# Dictionary Class with NLP methods
class Dictionary:
    def __init__(self, vocab: Union[List[str], Tuple[str]],
                 add_sos_token: bool = False, add_eos_token: bool = False):

        self.vocab = vocab
        self.add_sos_token = add_sos_token
        self.add_eos_token = add_eos_token
        self.punctuations = string.punctuation

        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.max_len = 0

        self._generateDictionary()
        if self.add_sos_token: self.max_len += 1
        if self.add_eos_token: self.max_len += 1

        self.vocab_vector = np.zeros((len(vocab), self.max_len), dtype=np.int32)
        self.vocab_vector + self.word2idx['<pad>']
        self._vectoriseVocab()

        del self.vocab
        gc.collect()

    def _generateDictionary(self):
        for sentence in self.vocab:
            self._addSentence(sentence)

    def _vectoriseVocab(self):
        for i, sentence in enumerate(self.vocab):
            sentence_vector = self.sentence2Vec(sentence)
            self.vocab_vector[i, :len(sentence_vector)] = sentence_vector

        if self.add_eos_token:
            self.vocab_vector[:, -1] = self.word2idx['<eos>']

    def sentence2Vec(self, sentence: str):
        vector = []
        sentence = self._tokenizeSentence(sentence)

        if self.add_sos_token: vector.append(self.word2idx['<sos>'])
        for word in sentence:
            word = self._removePunctuation(word)
            if word not in self.word2idx:
                vector.append(self.word2idx['<unk>'])
                continue
            vector.append(self.word2idx[word])
        return vector

    def vec2Sentence(self, vector: Union[np.ndarray, List[str], Tuple[str]]):
        assert len(vector.shape) == 1, 'vector must be 1-dimensional'
        sentence = [self.idx2word[i] for i in vector]
        sentence = ' '.join(sentence)
        for i in range(4):
            sentence = sentence.replace(self.idx2word[i], '')
        return sentence

    def _addSentence(self, sentence: str):
        sentence = self._tokenizeSentence(sentence)
        sent_len = len(sentence)
        if sent_len > self.max_len:
            self.max_len = sent_len
        for word in sentence:
            self._addWord(word)

    def _addWord(self, word: str):
        word = self._removePunctuation(word)
        if word not in self.word2idx:
            word_idx = 0 if len(self.idx2word) == 0 else max(self.idx2word) + 1
            self.word2idx[word] = word_idx
            self.idx2word[word_idx] = word

    def _tokenizeSentence(self, sentence: str):
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        return sentence

    def _removePunctuation(self, word: str):
        for i in word:
            if i in self.punctuations:
                word = word.replace(i, '')
        return word


# Dataset Class
class TextDataset(Dataset):
    def __init__(self, src: np.ndarray, tgt: np.ndarray):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.from_numpy(self.src[idx]).type(torch.IntTensor)
        tgt = torch.from_numpy(self.tgt[idx]).type(torch.IntTensor)
        return src, tgt


# Training and Testing Class
class FitterPipeline:
    def __init__(self, model, lossfunc, optimizer,
                 weight_init=True, custom_weight_initializer=None):

        self.model = model
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer

        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

    def xavier_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def save_model(self, dirname: str = './model_params',
                   filename='translation_model.pth.tar'):
        if not os.path.isdir(dirname): os.mkdir(dirname)
        state_dicts = {
            'model_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, os.path.join(dirname, filename))

    def train(self, dataloader: DataLoader, verbose: bool = False, device: str = 'cpu'):
        self.model.train()
        avg_loss = 0
        for i, (src, tgt) in tqdm.tqdm(enumerate(dataloader)):
            src = src.to(device)
            tgt = tgt.to(device)

            self.model.zero_grad()

            output = self.model(src, tgt[:, :-1])
            tgt = tgt[:, 1:]
            tgt = tgt.type(torch.LongTensor).reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            loss = self.lossfunc(output.cpu(), tgt)
            avg_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        avg_loss = avg_loss / (i + 1)
        PPL = math.exp(avg_loss)
        if verbose: print(f'training loss: {avg_loss:.3f} | training PPL: {PPL:7.3f}')
        return avg_loss, PPL

    def test(
            self, dataloader: DataLoader, verbose: bool = False, device: str = 'cpu'):
        self.model.eval()
        avg_loss = 0
        with torch.no_grad():
            for i, (src, tgt) in tqdm.tqdm(enumerate(dataloader)):
                src = src.to(device)
                tgt = tgt.to(device)
                output = self.model(src, tgt[:, :-1])
                tgt = tgt[:, 1:]
                tgt = tgt.type(torch.LongTensor).reshape(-1)
                output = output.reshape(-1, output.shape[-1])
                loss = self.lossfunc(output.cpu(), tgt)
                avg_loss += loss.item()
            avg_loss = avg_loss / (i + 1)
            PPL = math.exp(avg_loss)
            if verbose: print(f'testing loss: {avg_loss:.3f} | testing PPL: {PPL:7.3f}')
        return avg_loss, PPL

    def translate(self, src: torch.IntTensor, sos_token: int, target_len: int):
        device = next(self.model.parameters()).device
        src = src.to(device)
        batch_size, _ = src.shape
        translations = torch.zeros(batch_size, target_len)
        translations = translations.type(torch.IntTensor).to(device)
        translations[:, 0] = sos_token

        for i in tqdm.tqdm(range(1, target_len)):
            tgt = translations[:, 0:i]
            out = self.model(src, tgt)
            out = torch.argmax(out, dim=-1)
            translations[:, i] = out[:, -1]
        return translations[:, 1:]


#----------------------------------------------------------------------------------------------------------------------

def csv_2_Dictionary(csv_path: str, n_samples = 30000 ):  # 60000 - for part
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1)
    df = df.iloc[:n_samples, :]  # for part
    df.head()

    vv = df['French words/sentences'].values
    src_dictionary = Dictionary( vv )
    tgt_dictionary = Dictionary( df['English words/sentences'].values, add_sos_token=True, add_eos_token=True )

    return src_dictionary, tgt_dictionary


def parquet_opus_to_Dictionary( file_path, source_lang='en', target_lang='ru', n_samples = 30000 ): # Специальная версия для датасетов OPUS (как opus_books)
    pass