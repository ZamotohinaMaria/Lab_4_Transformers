#imports
import torch, string, gc, tqdm
import torch.nn as nn
import numpy as np
import time
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple, List
from sklearn.model_selection import train_test_split

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PosEmbedding(nn.Module):
    def __init__(self, h: int, padding_idx: int, n: int = 1000):
        super(PosEmbedding, self).__init__()
        self.h = h
        self.n = n
        self.padding_idx = padding_idx

    def forward(self, x: torch.IntTensor):
        assert len(x.shape) == 2, f'input must be 2 dimensional, {len(x.shape)} dimensions are given'
        N, L = x.shape

        output = torch.zeros(N, L, self.h, device=x.device)
        mask = torch.ones(N, L, self.h, device=x.device)
        mask[x == self.padding_idx] = 0

        dimensions = [i for i in range(self.h // 2)]

        for idx in range(L):
            for i in dimensions:
                val = x[:, idx] / (self.n ** (2 * i / self.h))
                output[:, idx, 2 * i] = torch.sin(val)
                output[:, idx, (2 * i) + 1] = torch.cos(val)

        output = output.masked_fill(mask == 0, 0)
        return output


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        self.sofmax = nn.Softmax(dim=-1)

    def forward(self,
                Q: Optional[torch.FloatTensor],
                K: Optional[torch.FloatTensor],
                V: Optional[torch.FloatTensor],
                padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):

        attn_energy = torch.matmul(Q, K.transpose(-2, -1))
        attn_energy /= np.sqrt(K.shape[-1])

        if torch.is_tensor(padding_mask):
            # shape: [N, seq_len]
            padding_mask = padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
            attn_energy = attn_energy.masked_fill(padding_mask == 0, -torch.inf)

        if torch.is_tensor(attention_mask):
            # shape: [N, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(dim=1)
            attn_energy = attn_energy.masked_fill(attention_mask == 0, -torch.inf)

        attn_energy = self.sofmax(attn_energy)
        output = torch.matmul(attn_energy, V)
        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert input_dim % n_heads == 0, 'input_dim must be divisible by n_head'

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.dropout = dropout
        self.head_dim = self.input_dim // self.n_heads

        self.Q_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.K_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.V_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.attention = DotProductAttention()
        self.fc = nn.Linear(input_dim, self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self,
                Q: Optional[torch.FloatTensor],
                K: Optional[torch.FloatTensor],
                V: Optional[torch.FloatTensor],
                padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):
        assert Q.shape[-1] % self.n_heads == 0, f'vector dimension of Q must be divisible by {self.n_head}'
        assert K.shape[-1] % self.n_heads == 0, f'vector dimension of K must be divisible by {self.n_head}'
        assert V.shape[-1] % self.n_heads == 0, f'vector dimension of V must be divisible by {self.n_head}'

        batch_size, _, _ = Q.shape

        Q = self.Q_fc(Q)
        K = self.K_fc(K)
        V = self.V_fc(V)

        Q = Q.reshape(batch_size, Q.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, K.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, V.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = self.attention(Q, K, V, padding_mask, attention_mask)
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(batch_size, -1, self.input_dim)

        output = self.fc(attn)
        output = self.dropout_layer(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        self.norm1 = nn.LayerNorm(self.input_dim)

        self.pointwise_ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.input_dim)
        )
        self.norm2 = nn.LayerNorm(self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.FloatTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None):
        attn = self.self_attention(x, x, x, src_padding_mask)
        x = self.norm1(x + attn)

        output = self.pointwise_ffn(x)
        output += x
        output = self.norm2(output)
        output = self.dropout_layer(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self,
                 model_dim: int, n_encoders: int,
                 src_vocab_size: int, padding_idx: Optional[int] = None,
                 n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.model_dim = model_dim
        self.n_encoders = n_encoders
        self.padding_idx = padding_idx
        self.src_vocab_size = src_vocab_size
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.word_embedding = nn.Embedding(
            self.src_vocab_size, self.model_dim, self.padding_idx)
        self.pos_embedding = PosEmbedding(self.model_dim, self.padding_idx)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.encoder_layers = nn.ModuleList(self.makeEncoderLayers())

    def forward(self,
                src: torch.IntTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None):
        word_embeddings = self.word_embedding(src)
        pos_embeddings = self.pos_embedding(src)

        output = pos_embeddings + word_embeddings
        for layers in self.encoder_layers:
            output = layers(output, src_padding_mask)

        output = self.dropout_layer(output)
        return output

    def makeEncoderLayers(self):
        return [
            TransformerEncoderLayer(
                self.model_dim, self.n_heads, self.dim_feedforward, self.dropout) \
            for i in range(self.n_encoders)
        ]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.cross_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        self.norm2 = nn.LayerNorm(self.input_dim)

        self.pointwise_ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.input_dim)
        )
        self.norm3 = nn.LayerNorm(self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self,
                x: torch.FloatTensor,
                encoder_output: torch.FloatTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None,
                tgt_padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):
        attn1 = self.self_attention(x, x, x, tgt_padding_mask, attention_mask)
        x = self.norm1(x + attn1)

        attn2 = self.cross_attention(
            x, encoder_output, encoder_output, src_padding_mask)
        x = self.norm2(x + attn2)

        output = self.pointwise_ffn(x)
        output += x
        output = self.norm2(output)
        output = self.dropout_layer(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self,
                 model_dim: int, n_decoders: int,
                 tgt_vocab_size: int, padding_idx: Optional[int] = None,
                 n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()

        self.model_dim = model_dim
        self.n_decoders = n_decoders
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.word_embedding = nn.Embedding(
            self.tgt_vocab_size, self.model_dim, self.padding_idx)
        self.pos_embedding = PosEmbedding(self.model_dim, self.padding_idx)
        self.decoder_layers = nn.ModuleList(self.makeDecoderLayers())

    def forward(self,
                tgt: torch.IntTensor,
                encoder_output: torch.FloatTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None,
                tgt_padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):
        word_embeddings = self.word_embedding(tgt)
        pos_embeddings = self.pos_embedding(tgt)

        output = word_embeddings + pos_embeddings
        for layers in self.decoder_layers:
            output = layers(
                output, encoder_output, src_padding_mask, tgt_padding_mask, attention_mask)

        return output

    def makeDecoderLayers(self):
        return [
            TransformerDecoderLayer(
                self.model_dim, self.n_heads, self.dim_feedforward, self.dropout)
            for i in range(self.n_decoders)
        ]


class Seq2seqTransformer(nn.Module):
    def __init__(self,
                 model_dim: int, n_encoders: int,
                 n_decoders: int, src_vocab_size: int,
                 tgt_vocab_size: int, src_padding_idx: Optional[int] = None,
                 tgt_padding_idx: Optional[int] = None, n_heads: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1, device: str = 'cpu'):
        super(Seq2seqTransformer, self).__init__()

        self.model_dim = model_dim
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device

        self.encoder = TransformerEncoder(
            self.model_dim, self.n_encoders, self.src_vocab_size,
            self.src_padding_idx, self.n_heads, self.dim_feedforward,
            self.dropout)

        self.decoder = TransformerDecoder(
            self.model_dim, self.n_decoders, self.tgt_vocab_size,
            self.tgt_padding_idx, self.n_heads, self.dim_feedforward,
            self.dropout)

        self.fc = nn.Linear(self.model_dim, self.tgt_vocab_size)

        self.to(self.device)

    def forward(self, src: torch.IntTensor, tgt: torch.IntTensor):
        batch_size, src_sequence_length = src.shape
        _, tgt_sequence_length = tgt.shape

        # src and tgt padding masks
        src_padding_mask = self.paddingMask(x=src, padding_idx=self.src_padding_idx)
        tgt_padding_mask = self.paddingMask(x=tgt, padding_idx=self.tgt_padding_idx)

        # tgt attention mask
        attention_mask = self.diagonalMask(
            batch_size, tgt_sequence_length, tgt_sequence_length)

        enc_output = self.encoder(src, src_padding_mask)
        dec_output = self.decoder(
            tgt, enc_output, src_padding_mask, tgt_padding_mask, attention_mask)

        dec_output = self.fc(dec_output)
        return dec_output

    def paddingMask(self, x: torch.IntTensor, padding_idx: int):
        padding_mask = torch.zeros(*x.shape, device=self.device)
        padding_mask[x != padding_idx] = 1
        return padding_mask

    def diagonalMask(self, *shape: int):
        diagonal_mask = torch.ones(*shape, device=self.device)
        diagonal_mask = torch.tril(diagonal_mask)
        return diagonal_mask




#***********************************************************************************************************************
#*******************************************  TRAINING AND TESTING  ****************************************************
#***********************************************************************************************************************

from transformer_notebook_helper_module import TextDataset, FitterPipeline, csv_2_Dictionary

start_time = time.time()

# choose required line bellow
src_dictionary, tgt_dictionary = csv_2_Dictionary('../input/en_fr/en_fr.csv', 30000)
# src_dictionary, tgt_dictionary = parquet_opus_to_Dictionary('../input/en_ru/opus_books_en_ru.parquet', "en", "ru" )  # dataset: https://huggingface.co/datasets/Helsinki-NLP/opus_books/tree/main

src = src_dictionary.vocab_vector
tgt = tgt_dictionary.vocab_vector

src_train, src_test, tgt_train, tgt_test = train_test_split(src, tgt, train_size=0.90)

train_dataset = TextDataset(src_train, tgt_train)
test_dataset = TextDataset(src_test, tgt_test)

batch_size = 256
n_workers = 0  # n_workers = 2

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, num_workers = n_workers)

test_dataloader = DataLoader(
    test_dataset, shuffle=True, batch_size=batch_size, num_workers = n_workers)


src_vocab_size = len(src_dictionary.word2idx)
tgt_vocab_size = len(tgt_dictionary.word2idx)

src_padding_idx = src_dictionary.word2idx['<pad>']
tgt_padding_idx = tgt_dictionary.word2idx['<pad>']

h_dim = 256 # 512
n_encoders = 1
n_decoders = 1
n_heads = 8
dropout = 0.10
dim_feedforward = 2048

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE =", DEVICE)



transformer = Seq2seqTransformer(
    h_dim,
    n_encoders=n_encoders,
    n_decoders=n_decoders,
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_padding_idx=src_padding_idx,
    tgt_padding_idx=tgt_padding_idx,
    n_heads=n_heads,
    dropout=dropout,
    dim_feedforward=dim_feedforward,
    device=DEVICE
)

lr = 5e-4
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
lossfunc = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.4) # , verbose=True
pipeline = FitterPipeline(transformer, lossfunc, optimizer)

EPOCHS = 10 # 14
train_losses, train_PPLs = [], []
test_losses, test_PPLs = [], []

best_loss = np.inf

for epoch in range(EPOCHS):
    print(f'epoch: {epoch}')
    print('training: ')
    train_loss, train_PPL = pipeline.train( train_dataloader, verbose=True, device=DEVICE)
    train_losses.append(train_loss)
    train_PPLs.append(train_PPL)

    print('testing: ')
    test_loss, test_PPL = pipeline.test(test_dataloader, verbose=True, device=DEVICE)
    test_losses.append(test_loss)
    test_PPLs.append(test_PPL)

    lr_scheduler.step()

    if test_loss < best_loss:
        best_loss = test_loss
        pipeline.save_model()
        print(f'model_saved at epoch: {epoch} | best_loss: {best_loss:.3f}')
    print('\n\n')


fig, axs = plt.subplots(1, 2, figsize=(20, 7))

axs[0].plot(train_losses, label='training loss')
axs[0].plot(test_losses, label='testing loss')
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss Plot')
axs[0].legend()

axs[1].plot(train_PPLs, label='training PPL')
axs[1].plot(test_PPLs, label='testing PPL')
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('PPL')
axs[1].set_title('Perplexity in Language')
axs[1].legend()


#***********************************************************************************************************************
#***********************************************************************************************************************

best_model_path = './model_params/translation_model.pth.tar'
best_model_state = torch.load(best_model_path)['model_params']

pipeline.model.load_state_dict(best_model_state)



def log_results(src_vectors, tgt_vectors):
    translated_vector = pipeline.translate(
        src_vectors,
        sos_token=src_dictionary.word2idx['<sos>'],
        target_len=src_dictionary.max_len)

    translated_vector = translated_vector.cpu().numpy()

    for idx in range(len(src_vectors)):
        model_translation = tgt_dictionary.vec2Sentence(translated_vector[idx])
        actual_translation = tgt_dictionary.vec2Sentence(tgt_vectors[idx])
        print(f'model translation: \n{model_translation}', '\n')
        print(f'actual translation: \n{actual_translation} \n')
        print('-'*90)



src_vectors = train_dataset[0:10][0]
tgt_vectors = train_dataset[0:10][1].numpy()

log_results(src_vectors, tgt_vectors)



src_vectors = test_dataset[0:10][0]
tgt_vectors = test_dataset[0:10][1].numpy()

log_results(src_vectors, tgt_vectors)


print(f"Длительность: {time.time() - start_time:.2f} секунд")



