import os
import sentencepiece as spm
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

def train_sentencepiece(cfg, is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    if is_src:
        train_file = f"{cfg.data_dir}/train.{cfg.src_lang}"
        model_prefix = f"{cfg.sp_dir}/{cfg.src_model_prefix}"
    else:
        train_file = f"{cfg.data_dir}/train.{cfg.tgt_lang}"
        model_prefix = f"{cfg.sp_dir}/{cfg.tgt_model_prefix}"

    print(f"===> Processing file: {train_file}")
    if not os.path.isdir(cfg.sp_dir):
        os.mkdir(cfg.sp_dir)

    sp_cfg = template.format(
        train_file,
        cfg.pad_id,
        cfg.sos_id,
        cfg.eos_id,
        cfg.unk_id,
        model_prefix,
        cfg.sp_vocab_size,
        cfg.character_coverage,
        cfg.model_type)

    spm.SentencePieceTrainer.Train(sp_cfg)

class NMTDataset(Dataset):
    def __init__(self, cfg, data_type="train"):
        super().__init__()
        self.cfg = cfg

        self.sp_src, self.sp_tgt = self.load_sp_tokenizer()
        self.src_texts, self.tgt_texts = self.read_data(data_type)

        src_tokenized_sequences = self.texts_to_sequences(self.src_texts, True)
        tgt_input_tokenized_sequences, tgt_output_tokenized_sequences = self.texts_to_sequences(self.tgt_texts, False)

        self.src_data = torch.LongTensor(src_tokenized_sequences)
        self.input_tgt_data = torch.LongTensor(tgt_input_tokenized_sequences)
        self.output_tgt_data = torch.LongTensor(tgt_output_tokenized_sequences)

    def read_data(self, data_type):
        print(f"===> Load data from: {self.cfg.data_dir}/{data_type}.{self.cfg.src_lang}")
        with open(f"{self.cfg.data_dir}/{data_type}.{self.cfg.src_lang}", 'r') as f:
            src_texts = f.readlines()

        print(f"===> Load data from: {self.cfg.data_dir}/{data_type}.{self.cfg.tgt_lang}")
        with open(f"{self.cfg.data_dir}/{data_type}.{self.cfg.tgt_lang}", 'r') as f:
            trg_texts = f.readlines()

        return src_texts, trg_texts

    def load_sp_tokenizer(self):
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(f"{self.cfg.sp_dir}/{self.cfg.src_model_prefix}.model")

        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.Load(f"{self.cfg.sp_dir}/{self.cfg.tgt_model_prefix}.model")

        return sp_src, sp_tgt

    def texts_to_sequences(self, texts, is_src=True):
        if is_src:
            src_tokenized_sequences = []
            for text in tqdm(texts):
                tokenized = self.sp_src.EncodeAsIds(text.strip())
                src_tokenized_sequences.append(
                    pad_or_truncate([self.cfg.sos_id] + tokenized + [self.cfg.eos_id], self.cfg.seq_len, self.cfg.pad_id)
                )
            return src_tokenized_sequences
        else:
            tgt_input_tokenized_sequences = []
            tgt_output_tokenized_sequences = []
            for text in tqdm(texts):
                tokenized = self.sp_tgt.EncodeAsIds(text.strip())
                tgt_input = [self.cfg.sos_id] + tokenized
                tgt_output = tokenized + [self.cfg.eos_id]
                tgt_input_tokenized_sequences.append(pad_or_truncate(tgt_input, self.cfg.seq_len, self.cfg.pad_id))
                tgt_output_tokenized_sequences.append(pad_or_truncate(tgt_output, self.cfg.seq_len, self.cfg.pad_id))

            return tgt_input_tokenized_sequences, tgt_output_tokenized_sequences

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_tgt_data[idx], self.output_tgt_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]

def pad_or_truncate(tokenized_sequence, seq_len, pad_id):
    if len(tokenized_sequence) < seq_len:
        left = seq_len - len(tokenized_sequence)
        padding = [pad_id] * left
        tokenized_sequence += padding
    else:
        tokenized_sequence = tokenized_sequence[:seq_len]
    return tokenized_sequence

def get_data_loader(cfg, data_type='train'):
    dataset = NMTDataset(cfg, data_type)

    if data_type == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)

    return dataset, dataloader