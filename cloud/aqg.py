#!/usr/bin/env python3

from model import *

aqg_tok = aqg_tokenizer()
aqg_mod = aqg_model()

def perform_aqg(keyword_array, corrected):
    questions = []

    for kw in keyword_array:
        inputs = f"Answer: {kw} | Context: {corrected}"
        input_ids = aqg_tok(inputs, return_tensors='pt', max_length=1024, truncation=True).input_ids
        outputs = aqg_mod.generate(input_ids, max_length=256, do_sample=True, top_p=0.98, temperature=0.9)
        test = aqg_tok.decode(outputs[0], skip_special_tokens=True)
        questions.append(f"{test} ({kw})")

    return questions
