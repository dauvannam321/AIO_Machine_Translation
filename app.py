import torch
import numpy as np
import datasets
import sentencepiece as spm
from Trainer import Trainer
from flask import Flask, render_template, request

class BaseConfig:
    """ base Encoder Decoder config """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class NMTConfigVI_EN(BaseConfig):
    # Dataset
    data_dir = 'vi_en/data'
    src_lang = 'vi'
    tgt_lang = 'en'
    # data_dir = 'en_vi/data'
    # src_lang = 'en'
    # tgt_lang = 'vi'

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

class NMTConfigEN_VI(BaseConfig):
    # Dataset
    data_dir = 'en_vi/data'
    src_lang = 'en'
    tgt_lang = 'vi'
    # data_dir = 'en_vi/data'
    # src_lang = 'en'
    # tgt_lang = 'vi'

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
    ckpt_dir = './en_vi'
    ckpt_name = 'best_ckpt.tar'

cfg_VI_EN = NMTConfigVI_EN()
vi_en_model = Trainer(cfg_VI_EN, is_train=False, load_ckpt=False)

cfg_EN_VI = NMTConfigVI_EN()
en_vi_model = Trainer(cfg_VI_EN, is_train=False, load_ckpt=False)

app = Flask(__name__)

switch_flag = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global switch_flag
    global vi_en_model
    global en_vi_model    

    if request.method == 'POST':
        if 'translate_button' in request.form:
            user_input = request.form['text_input']
            if not switch_flag:
                model = vi_en_model
            else:
                model = en_vi_model
            return render_template('index.html', input_text=model.inference(user_input))
        
        elif 'switch_button' in request.form:
            switch_flag = not switch_flag
            switch_button = switch_flag
            return render_template('index.html', input_text=None, switch_button=switch_button)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port="5005",debug=True)
