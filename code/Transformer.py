#imports
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)
torch.backends.cudnn.deterministic = True # заставляет cuDNN выбирать детерминированные алгоритмы (одинаковый результат при одинаковом входе).
torch.backends.cudnn.benchmark = False # отключает авто-подбор “самого быстрого” алгоритма cuDNN, который может давать недетерминированность.

# модуль позиционного кодирования
# фактически это не чистое позиционное кодирование, а кодирование, зависящее от значения токена в позиции.
class PosEmbedding(nn.Module):
    def __init__(self, h: int, padding_idx: int, n: int = 1000):
        super(PosEmbedding, self).__init__()
        self.h = h # размерность эмбеддинга (hidden size).
        self.n = n # база в формуле синусов/косинусов (аналог 10000 в классической статье, здесь 1000)
        self.padding_idx = padding_idx # индекс токена <pad>

    # прямой проход
    def forward(self, x: torch.IntTensor):
        assert len(x.shape) == 2, f'input must be 2 dimensional, {len(x.shape)} dimensions are given'
        
        # N — batch size.
        # L — длина последовательности.
        N, L = x.shape

        output = torch.zeros(N, L, self.h, device=x.device)
        mask = torch.ones(N, L, self.h, device=x.device) # маска для зануления pad-позиций.
        mask[x == self.padding_idx] = 0 # там, где токен pad, все h компонент маски ставятся в 0.
        
        # Формирование частот
        dimensions = [i for i in range(self.h // 2)]

        # Реализация формулы 14 стр 8
        for idx in range(L):
            for i in dimensions:
                val = x[:, idx] / (self.n ** (2 * i / self.h))
                output[:, idx, 2 * i] = torch.sin(val)
                output[:, idx, (2 * i) + 1] = torch.cos(val)

        output = output.masked_fill(mask == 0, 0) # для <pad> вектор обнуляется.
        return output # возвращается output формы (N, L, h).

# реализация Scaled Dot-Product Attention с двумя типами масок.
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
        # Считается матрица “схожести” Q и K:
        # attn_energy = Q @ K^T
        attn_energy = torch.matmul(Q, K.transpose(-2, -1))
        # Масштабирование: attn_energy /= sqrt(d_k)
        # нужно для стабильности softmax (чтобы значения не были слишком большими).
        attn_energy /= np.sqrt(K.shape[-1])

        # padding_mask (маска паддинга):
        # входная форма [N, seq_len].
        # расширяется до [N, 1, 1, seq_len].
        # где маска 0 (pad), ставится -inf:
        # после softmax эти позиции получат вес 0.
        if torch.is_tensor(padding_mask):
            # shape: [N, seq_len]
            padding_mask = padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
            attn_energy = attn_energy.masked_fill(padding_mask == 0, -torch.inf)

        # attention_mask (например, causal mask для декодера):
        # входная форма [N, seq_len, seq_len].
        # расширяется до [N, 1, seq_len, seq_len].
        # запрещенные позиции тоже получают -inf.
        if torch.is_tensor(attention_mask):
            # shape: [N, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(dim=1)
            attn_energy = attn_energy.masked_fill(attention_mask == 0, -torch.inf)

        attn_energy = self.sofmax(attn_energy)
        output = torch.matmul(attn_energy, V) # Взвешенная сумма V
        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert input_dim % n_heads == 0, 'input_dim must be divisible by n_head'

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.dropout = dropout
        # размер одной головы
        self.head_dim = self.input_dim // self.n_heads
        
        # Линейные проекции - обучаемые матрицы, которые строят новые Q,K,V перед attention.
        # nn.Linear — это полносвязный линейный слой. Вычисляет y = xW^T + b
        self.Q_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.K_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.V_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.attention = DotProductAttention() # SDPA
        # финальная “склейка” после объединения голов.
        self.fc = nn.Linear(input_dim, self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self,
                Q: Optional[torch.FloatTensor],
                K: Optional[torch.FloatTensor],
                V: Optional[torch.FloatTensor],
                padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):
        assert Q.shape[-1] % self.n_heads == 0, f'vector dimension of Q must be divisible by {self.n_heads}'
        assert K.shape[-1] % self.n_heads == 0, f'vector dimension of K must be divisible by {self.n_heads}'
        assert V.shape[-1] % self.n_heads == 0, f'vector dimension of V must be divisible by {self.n_heads}'

        batch_size, _, _ = Q.shape

        # Форма: (N, L, input_dim)
        Q = self.Q_fc(Q)
        K = self.K_fc(K)
        V = self.V_fc(V)

        # разбиение на головы
        # Q -> (N, n_heads, Lq, head_dim)
        # K -> (N, n_heads, Lk, head_dim)
        # V -> (N, n_heads, Lk, head_dim)
        # permute в — это перестановка осей тензора.
        Q = Q.reshape(batch_size, Q.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, K.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, V.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # применяем SDPA
        attn = self.attention(Q, K, V, padding_mask, attention_mask) 
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(batch_size, -1, self.input_dim)

        # финальная проекция
        output = self.fc(attn)
        output = self.dropout_layer(output)
        return output

# реализация одного блока энкодера
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward # ширина скрытого слоя FFN
        self.dropout = dropout

        # SDPA, где Q=K=V=x
        self.self_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        # нормализация после residual-суммы x + attn
        self.norm1 = nn.LayerNorm(self.input_dim)

        # позиционно-независимая FFN - Часть 6 стр 9:
        # Linear(input_dim -> dim_feedforward)
        # ReLU
        # Linear(dim_feedforward -> input_dim)
        # применяется к каждому токену отдельно с одинаковыми весами.
        self.pointwise_ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.input_dim)
        )
        # нормализация после residual-суммы x + ffn(x)
        self.norm2 = nn.LayerNorm(self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    # x формы (N, L, D):
    # N — batch size
    # L — длина последовательности
    # D = input_dim

    # src_padding_mask формы (N, L):
    # 1 для реальных токенов,
    # 0 для <pad>.
    def forward(self, x: torch.FloatTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None):
        attn = self.self_attention(x, x, x, src_padding_mask) # форма (N, L, D)
        x = self.norm1(x + attn)

        output = self.pointwise_ffn(x)
        output += x
        output = self.norm2(output)
        output = self.dropout_layer(output)
        return output # тензор (N, L, D)

# стек из n_encoders + эмбеддинги.
class TransformerEncoder(nn.Module):
    def __init__(self,
                 model_dim: int, n_encoders: int,
                 src_vocab_size: int, padding_idx: Optional[int] = None,
                 n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.model_dim = model_dim #D, размер эмбеддинга/скрытого состояния.
        self.n_encoders = n_encoders #
        self.padding_idx = padding_idx 
        self.src_vocab_size = src_vocab_size 
        self.n_heads = n_heads 
        self.dim_feedforward = dim_feedforward 
        self.dropout = dropout 

        # переводит индексы токенов в векторы (N, L, D)
        self.word_embedding = nn.Embedding(
            self.src_vocab_size, self.model_dim, self.padding_idx)
        # позиционные признаки той же размерности (N, L, D).
        self.pos_embedding = PosEmbedding(self.model_dim, self.padding_idx)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # список из n_encoders объектов TransformerEncoderLayer.
        # ModuleList нужен, чтобы PyTorch регистрировал параметры всех слоев.
        self.encoder_layers = nn.ModuleList(self.makeEncoderLayers())

    # Вход:
    # src формы (N, L) — индексы слов.
    # src_padding_mask формы (N, L).
    
    # Псевдокод 6.2 стр 10
    def forward(self,
                src: torch.IntTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None):
        word_embeddings = self.word_embedding(src) # Форма (N, L, D)
        pos_embeddings = self.pos_embedding(src) # Форма (N, L, D)

        
        output = pos_embeddings + word_embeddings # Формула 13 стр 8
        for layers in self.encoder_layers:
            output = layers(output, src_padding_mask)

        output = self.dropout_layer(output)
        return output
    
    # строит n_encoders одинаково параметризованных блоков.
    def makeEncoderLayers(self):
        return [
            TransformerEncoderLayer(
                self.model_dim, self.n_heads, self.dim_feedforward, self.dropout) \
            for i in range(self.n_encoders)
        ]

# один базовый слой декодера
class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.input_dim = input_dim # размер скрытого вектора токена
        self.n_heads = n_heads 
        self.dim_feedforward = dim_feedforward # ширина скрытого слоя FFN 
        self.dropout = dropout 

        # реализация схемы на стр 11
        # masked self-attention декодера.
        self.self_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        # LayerNorm после residual-суммы y + А.
        self.norm1 = nn.LayerNorm(self.input_dim)
        
        # cross-attention: запросы из декодера, ключи/значения из энкодера.
        self.cross_attention = MultiHeadedAttention(self.n_heads, self.input_dim, self.dropout)
        # LayerNorm после residual-суммы y~ + B.
        self.norm2 = nn.LayerNorm(self.input_dim)

        # FFN 
        # Linear(input_dim -> dim_feedforward)
        # ReLU
        # Linear(dim_feedforward -> input_dim)
        self.pointwise_ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.input_dim)
        )
        
        # LayerNorm после residual-суммы y~ + z~.
        self.norm3 = nn.LayerNorm(self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    # Входы:
    # x — текущее состояние декодера (N, Lt, D).
    # encoder_output — выход энкодера (N, Ls, D).
    # src_padding_mask — маска паддинга source (N, Ls).
    # tgt_padding_mask — маска паддинга target (N, Lt).
    # attention_mask — causal mask (N, Lt, Lt) (запрет смотреть в будущее).
    def forward(self,
                x: torch.FloatTensor,
                encoder_output: torch.FloatTensor,
                src_padding_mask: Optional[torch.FloatTensor] = None,
                tgt_padding_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None):
        # masked self-attention
        attn1 = self.self_attention(x, x, x, tgt_padding_mask, attention_mask)
        x = self.norm1(x + attn1)

        # cross-attention к энкодеру
        attn2 = self.cross_attention(
            x, encoder_output, encoder_output, src_padding_mask)
        x = self.norm2(x + attn2)

        # FFN
        output = self.pointwise_ffn(x)
        output += x
        output = self.norm3(output)
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
        word_embeddings = self.word_embedding(tgt) # Форма (N, L, D)
        pos_embeddings = self.pos_embedding(tgt) # Форма (N, L, D)

        output = word_embeddings + pos_embeddings # Формула 13 стр 8
        for layers in self.decoder_layers:
            output = layers(
                output, encoder_output, src_padding_mask, tgt_padding_mask, attention_mask)
        
        # Lt — это длина target-последовательности (число токенов в декодере).
        return output # форма (N, Lt, D)

    def makeDecoderLayers(self):
        return [
            TransformerDecoderLayer(
                self.model_dim, self.n_heads, self.dim_feedforward, self.dropout)
            for i in range(self.n_decoders)
        ]

# Это верхнеуровневая seq2seq-модель Transformer: объединяет encoder + decoder + проекцию в словарь.
# принимает source и target последовательности индексов,
# строит нужные маски,
# прогоняет encoder и decoder,
# выдает логиты по словарю target.
class Seq2seqTransformer(nn.Module):
    def __init__(self,
                 model_dim: int, n_encoders: int,
                 n_decoders: int, src_vocab_size: int,
                 tgt_vocab_size: int, src_padding_idx: Optional[int] = None,
                 tgt_padding_idx: Optional[int] = None, n_heads: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1, device: str = 'cpu'):
        super(Seq2seqTransformer, self).__init__()

        self.model_dim = model_dim # размер скрытого пространства D
        self.n_encoders = n_encoders # число слоев энкодера.
        self.n_decoders = n_decoders # число слоев декодера.
        self.src_vocab_size = src_vocab_size # размер source словаря.
        self.tgt_vocab_size = tgt_vocab_size #  размер target словаря.
        self.src_padding_idx = src_padding_idx # индекс <pad> для source.
        self.tgt_padding_idx = tgt_padding_idx # индекс <pad> для target.
        self.n_heads = n_heads # число голов в MHA
        self.dim_feedforward = dim_feedforward # ширина FFN внутри слоев.
        self.dropout = dropout
        self.device = device # cpu/cuda

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