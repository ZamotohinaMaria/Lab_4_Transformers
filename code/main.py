import os
import csv
import time
import sys
import importlib.util
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformer_notebook_helper_module import (
    TextDataset, FitterPipeline, csv_2_Dictionary, parquet_opus_to_Dictionary
)


# Training/data config
N_SAMPLES = 30000
RUN_TRAINING = True
RUN_INFERENCE = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODEL_PARAMS_DIR = os.path.join(PROJECT_ROOT, 'model_params')

# Which architecture file to use:
# 'Transformer.py'                     - base/original
# 'Transformer-a-GeLU.py'              - point a
# 'Transformer-b-LN_pre_post.py'       - point b
# 'Transformer-c-neiron_count.py'      - point c
# 'Transformer-d-MHA_head_count.py'    - point d
TRANSFORMER_FILE = 'Transformer.py'

# DATASET_KIND: 'csv' or 'parquet'
DATASET_KIND = 'csv'
DATASET_PATH = os.path.join(PROJECT_ROOT, 'input', 'en_fr', 'en_fr.csv')
# DATASET_PATH = os.path.join(PROJECT_ROOT, 'input', 'en_ru', 'opus_books_en_ru.parquet')
# DATASET_PATH = os.path.join(PROJECT_ROOT, 'input', 'de_en', 'opus_books_de_en.parquet')
PARQUET_SOURCE_LANG = 'en'
PARQUET_TARGET_LANG = 'ru'

TRAIN_SIZE = 0.90
BATCH_SIZE = 256
N_WORKERS = 0

# Model hyperparameters
H_DIM = 256
N_ENCODERS = 1
N_DECODERS = 1
N_HEADS = 8
DROPOUT = 0.10
DIM_FEEDFORWARD = 2048

# Training hyperparameters
LR = 5e-4
EPOCHS = 10
LR_STEP_SIZE = 10
LR_GAMMA = 0.4

def load_seq2seq_transformer_class(transformer_filename: str):
    transformer_path = os.path.join(SCRIPT_DIR, transformer_filename)
    if not os.path.isfile(transformer_path):
        raise FileNotFoundError(f'Transformer file not found: {transformer_path}')

    module_name = f"transformer_variant_{os.path.splitext(transformer_filename)[0]}"
    spec = importlib.util.spec_from_file_location(module_name, transformer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'Seq2seqTransformer'):
        raise AttributeError(
            f"File {transformer_filename} does not contain Seq2seqTransformer class"
        )
    return module.Seq2seqTransformer


def build_best_model_path(run_basename: str):
    filename = f'model_{run_basename}.pth.tar'
    return os.path.join(MODEL_PARAMS_DIR, filename)


def get_dataset_folder_name(dataset_path: str):
    return os.path.basename(os.path.dirname(os.path.normpath(dataset_path)))


def build_run_basename(
        dataset_path: str, h_dim: int, n_heads: int, dim_feedforward: int,
        transformer_file: str, run_dt: datetime = None):
    dataset_folder = get_dataset_folder_name(dataset_path)
    transformer_tag = os.path.splitext(transformer_file)[0]
    timestamp = (run_dt or datetime.now()).strftime("%H-%M_%d-%m-%Y")
    return (
        f'hdim{h_dim}_heads{n_heads}_ffn{dim_feedforward}'
        f'_run_{transformer_tag}_{dataset_folder}_{timestamp}'
    )


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_training_metrics(
        train_losses, test_losses, train_ppls, test_ppls,
        run_basename: str,
        results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    filename = f'{run_basename}.csv'
    output_path = os.path.join(results_dir, filename)

    with open(output_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_ppl', 'test_ppl'])
        for epoch_idx in range(len(train_losses)):
            writer.writerow([
                epoch_idx + 1,
                train_losses[epoch_idx],
                test_losses[epoch_idx],
                train_ppls[epoch_idx],
                test_ppls[epoch_idx]
            ])
    print(f'training metrics saved to: {output_path}')
    return output_path


def load_dictionaries():
    if DATASET_KIND == 'csv':
        return csv_2_Dictionary(DATASET_PATH, N_SAMPLES)
    if DATASET_KIND == 'parquet':
        return parquet_opus_to_Dictionary(
            DATASET_PATH, PARQUET_SOURCE_LANG, PARQUET_TARGET_LANG, N_SAMPLES
        )
    raise ValueError("DATASET_KIND must be 'csv' or 'parquet'")


def log_results(pipeline, src_dictionary, tgt_dictionary, src_vectors, tgt_vectors):
    translated_vector = pipeline.translate(
        src_vectors,
        sos_token=src_dictionary.word2idx['<sos>'],
        target_len=src_dictionary.max_len
    )
    translated_vector = translated_vector.cpu().numpy()

    for idx in range(len(src_vectors)):
        model_translation = tgt_dictionary.vec2Sentence(translated_vector[idx])
        actual_translation = tgt_dictionary.vec2Sentence(tgt_vectors[idx])
        print(f'model translation: \n{model_translation}', '\n')
        print(f'actual translation: \n{actual_translation} \n')
        print('-' * 90)


def main(run_basename: str):
    start_time = time.time()
    seq2seq_cls = load_seq2seq_transformer_class(TRANSFORMER_FILE)
    best_model_path = build_best_model_path(run_basename)
    print(f"TRANSFORMER_FILE = {TRANSFORMER_FILE}")

    src_dictionary, tgt_dictionary = load_dictionaries()
    src = src_dictionary.vocab_vector
    tgt = tgt_dictionary.vocab_vector

    src_train, src_test, tgt_train, tgt_test = train_test_split(
        src, tgt, train_size=TRAIN_SIZE
    )

    train_dataset = TextDataset(src_train, tgt_train)
    test_dataset = TextDataset(src_test, tgt_test)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS
    )

    src_vocab_size = len(src_dictionary.word2idx)
    tgt_vocab_size = len(tgt_dictionary.word2idx)
    src_padding_idx = src_dictionary.word2idx['<pad>']
    tgt_padding_idx = tgt_dictionary.word2idx['<pad>']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("DEVICE =", device)

    transformer = seq2seq_cls(
        H_DIM,
        n_encoders=N_ENCODERS,
        n_decoders=N_DECODERS,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_padding_idx=src_padding_idx,
        tgt_padding_idx=tgt_padding_idx,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        dim_feedforward=DIM_FEEDFORWARD,
        device=device
    )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
    lossfunc = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, LR_STEP_SIZE, gamma=LR_GAMMA
    )
    pipeline = FitterPipeline(transformer, lossfunc, optimizer)

    train_losses, train_ppls = [], []
    test_losses, test_ppls = [], []
    best_loss = np.inf

    if RUN_TRAINING:
        for epoch in range(EPOCHS):
            print(f'epoch: {epoch}')
            print('training: ')
            train_loss, train_ppl = pipeline.train(train_dataloader, verbose=True, device=device)
            train_losses.append(train_loss)
            train_ppls.append(train_ppl)

            print('testing: ')
            test_loss, test_ppl = pipeline.test(test_dataloader, verbose=True, device=device)
            test_losses.append(test_loss)
            test_ppls.append(test_ppl)

            lr_scheduler.step()

            if test_loss < best_loss:
                best_loss = test_loss
                model_name = os.path.basename(best_model_path)
                pipeline.save_model(dirname=MODEL_PARAMS_DIR, filename=model_name)
                print(f'model_saved at epoch: {epoch} | best_loss: {best_loss:.3f}')
            print('\n\n')

        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
        axs[0].plot(train_losses, label='training loss')
        axs[0].plot(test_losses, label='testing loss')
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss Plot')
        axs[0].legend()

        axs[1].plot(train_ppls, label='training PPL')
        axs[1].plot(test_ppls, label='testing PPL')
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('PPL')
        axs[1].set_title('Perplexity in Language')
        axs[1].legend()
        plt.tight_layout()

        save_training_metrics(
            train_losses, test_losses, train_ppls, test_ppls,
            run_basename
        )
    else:
        print('RUN_TRAINING=False: training skipped.')
        save_training_metrics(
            train_losses, test_losses, train_ppls, test_ppls,
            run_basename
        )

    if RUN_INFERENCE:
        if os.path.isfile(best_model_path):
            best_model_state = torch.load(best_model_path)['model_params']
            pipeline.model.load_state_dict(best_model_state)
        else:
            raise FileNotFoundError(
                f"Model file is not found: {best_model_path}. "
                "Train model first or set RUN_INFERENCE=False."
            )

        src_vectors = train_dataset[0:10][0]
        tgt_vectors = train_dataset[0:10][1].numpy()
        log_results(pipeline, src_dictionary, tgt_dictionary, src_vectors, tgt_vectors)

        src_vectors = test_dataset[0:10][0]
        tgt_vectors = test_dataset[0:10][1].numpy()
        log_results(pipeline, src_dictionary, tgt_dictionary, src_vectors, tgt_vectors)

    print(f"Duration: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    run_basename = build_run_basename(
        DATASET_PATH, H_DIM, N_HEADS, DIM_FEEDFORWARD, TRANSFORMER_FILE
    )
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f'{run_basename}.log')

    with open(log_path, mode='w', encoding='utf-8', buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f'log file: {log_path}')
            main(run_basename)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
