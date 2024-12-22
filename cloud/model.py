from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline, T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT

import torch

"""
File name: model.py
Usage: Contains functions that return the needed models
"""

device = torch.device('cpu')

trocr_model_dir = './trocr-large-handwritten'
kw_model_name = 'all-MiniLM-L6-v2'
syntax_model_dir = './syntax-model'
aqg_model_dir = './aqg-t5'

def trocr_processor():
    return TrOCRProcessor.from_pretrained(trocr_model_dir)

def trocr_model():
    return VisionEncoderDecoderModel.from_pretrained(trocr_model_dir).to(device)

def kw_model():
    return KeyBERT(model=kw_model_name)

def syntax_model():
    return T5ForConditionalGeneration.from_pretrained(syntax_model_dir).to(device)

def syntax_tokenizer():
    return T5Tokenizer.from_pretrained(syntax_model_dir)

def aqg_model():
    return T5ForConditionalGeneration.from_pretrained(aqg_model_dir).to(device)

def aqg_tokenizer():
    return T5Tokenizer.from_pretrained(aqg_model_dir)
