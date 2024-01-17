import torch
from Trainer import Trainer

class BaseConfig:
    """ base Encoder Decoder config """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class NMTConfig(BaseConfig):
    # Dataset
    data_dir = 'vi_en/data'
    src_lang = 'vi'
    tgt_lang = 'en'

    # Tokenizer
    sp_dir = data_dir + '/sp'
    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    src_model_prefix = 'sp_' + src_lang
    tgt_model_prefix = 'sp_' + tgt_lang
    sp_vocab_size = 10000
    character_coverage = 1.0
    model_type = 'unigram'

    # Model
    num_heads = 8
    num_layers = 6
    d_model = 512
    d_ff = 2048
    drop_out = 0.1

    # Training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learning_rate = 1e-4
    batch_size = 64
    seq_len = 150
    num_epochs = 1
    ckpt_dir = './vi_en'
    ckpt_name = 'best_ckpt.tar'

cfg = NMTConfig()
trainer = Trainer(cfg, is_train=False, load_ckpt=False)
trainer.inference('Xin chào.')