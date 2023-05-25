#!/usr/bin/env python3

'''
@Time   : 23 An 2020
@Author : Oguz, Cennet
@Desc   : 
'''

import torch
import torch.nn as nn
import itertools
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


'''
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
'''
MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    # 'xlnet': (XLNetModel, XLNetTokenizer),
}

# model_path = "../bert_finetune/bert_large_snips"


def prepare_inputs_for_bert_xlnet(sentences, word_lengths, tokenizer, cls_token_at_end=False,
                                  pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
                                  pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1,
                                  pad_token_segment_id=0, device=None):
    max_length_of_sentences = max(word_lengths)
    tokens, segment_ids, selected_indexes = [], [], []
    tokenized_texts, word_piece_indexes = [], []
    start_pos = 0
    for ws in sentences:
        selected_index, word_piece_index = [], []
        ts = []
        for w in ws:
            stem_idx = len(ts) + 1
            if cls_token_at_end:
                stem_idx -= 1
            selected_index.append(stem_idx)
            tokenized_word = tokenizer.tokenize(w)
            if len(tokenized_word) > 1:
                word_piece_index.append(
                    [i for i in range(stem_idx, stem_idx + len(tokenized_word))])
            ts += tokenized_word
        ts += [sep_token]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
        word_piece_indexes.append(word_piece_index)
    max_length_of_tokens = max([len(tokenized_text)
                                for tokenized_text in tokens])
    tokenized_texts = [tokenized_text for tokenized_text in tokens]
    padding_lengths = [max_length_of_tokens -
                       len(tokenized_text) for tokenized_text in tokens]

    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] *
                      len(tokenized_text) for idx, tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(
            tokenized_text) for idx, tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] +
                        si for idx, si in enumerate(segment_ids)]
        selected_indexes = [[padding_lengths[idx] + i + idx * max_length_of_tokens for i in selected_index]
                            for idx, selected_index in enumerate(selected_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx]
                      for idx, tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(
            tokenized_text) + [pad_token] * padding_lengths[idx] for idx, tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx]
                        for idx, si in enumerate(segment_ids)]
        selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index]
                            for idx, selected_index in enumerate(selected_indexes)]
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(
        length)] for idx, length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(
        indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(
        segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(
        selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(
        copied_indexes)), dtype=torch.long, device=device)
    return {'tokens': tokens_tensor,
            'segments': segments_tensor,
            'selects': selects_tensor,
            'copies': copies_tensor,
            'mask': input_mask,
            'tokenized_text': tokenized_texts,
            'word_piece_indexes': word_piece_indexes}


def sum_wordpieces(item, indexe_lists):
    for index_list in indexe_lists:
        first = index_list[0]
        last = index_list[-1] + 1
        item[index_list[0]] = torch.sum(item[first:last], dim=0)
    return item


class bert_embeddings():
    def __init__(self, model_type='bert', model_name='bert-large-cased', device=None):
        self.model_type = model_type
        self.model_name = model_name
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        #############Fine-tuned BERT##########################
        #self.config = config_class.from_pretrained(model_path)
        #self.tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
        #self.model = model_class.from_pretrained(model_path, config=self.config)
        #############No fine-tuning BERT##########################
        self.config = config_class.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True)
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(
            model_name, config=self.config)

        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # print(self.model.config)
        print('Loaded bert model')

    def get_vectors(self, words):
        lengths = [len(s) for s in words]
        max_len = max(lengths)
        mask = torch.Tensor([[1]*len(x_) + [0]*(max_len-len(x_))
                             for x_ in words]).long().to(device=self.device)
        with torch.no_grad():
            inputs = prepare_inputs_for_bert_xlnet(words, lengths, self.tokenizer,
                                                   # xlnet has a cls token at the end
                                                   cls_token_at_end=bool(
                                                       self.model_type in ['xlnet']),
                                                   cls_token=self.tokenizer.cls_token,
                                                   sep_token=self.tokenizer.sep_token,
                                                   cls_token_segment_id=2 if self.model_type in [
                                                       'xlnet'] else 0,
                                                   # pad on the left for xlnet
                                                   pad_on_left=bool(
                                                       self.model_type in ['xlnet']),
                                                   pad_token_segment_id=4 if self.model_type in [
                                                       'xlnet'] else 0,
                                                   device=self.device)
            tokens, segments, selects, copies, attention_mask = inputs['tokens'], inputs[
                'segments'], inputs['selects'], inputs['copies'], inputs['mask']
            tokenized_text, word_piece_indexes = inputs['tokenized_text'], inputs['word_piece_indexes']
            outputs = self.model(
                tokens, token_type_ids=segments, attention_mask=attention_mask)
            pretrained_hiddens = outputs[2]
            '''
            #################### HIDDENS ########################
            Shape: batch_size, sequence_length, hidden_size
            '''
            h1 = pretrained_hiddens[-10]
            h2 = pretrained_hiddens[-11]
            h3 = pretrained_hiddens[-12]
            h4 = pretrained_hiddens[-13]

            '''
            for i in range(0, h1.shape[0]):
                h1[i] = sum_wordpieces(h1[i], word_piece_indexes[i])
                h2[i] = sum_wordpieces(h2[i], word_piece_indexes[i])
                h3[i] = sum_wordpieces(h3[i], word_piece_indexes[i])
                h4[i] = sum_wordpieces(h4[i], word_piece_indexes[i])
            '''

            batch_size, pretrained_seq_length, hidden_size = h1.size(
                0), h1.size(1), h1.size(2)
            h1_chosen_hiddens = h1.view(-1,
                                        hidden_size).index_select(0, selects)
            h2_chosen_hiddens = h2.view(-1,
                                        hidden_size).index_select(0, selects)
            h3_chosen_hiddens = h3.view(-1,
                                        hidden_size).index_select(0, selects)
            h4_chosen_hiddens = h4.view(-1,
                                        hidden_size).index_select(0, selects)

            embeds = torch.zeros(len(lengths) * max(lengths),
                                 hidden_size, device=self.device)
            h1_embeds = embeds.index_copy_(0, copies, h1_chosen_hiddens).view(
                len(lengths), max(lengths), -1)
            h2_embeds = embeds.index_copy_(0, copies, h2_chosen_hiddens).view(
                len(lengths), max(lengths), -1)
            h3_embeds = embeds.index_copy_(0, copies, h3_chosen_hiddens).view(
                len(lengths), max(lengths), -1)
            h4_embeds = embeds.index_copy_(0, copies, h4_chosen_hiddens).view(
                len(lengths), max(lengths), -1)

            concat_hiddens = torch.cat(
                (h1_embeds, h2_embeds, h3_embeds, h4_embeds), dim=2)

            # print(concat_hiddens.shape)

        return concat_hiddens, mask


if __name__ == "__main__":

    to_get_bert_embeddings = bert_embeddings()

    # use batch_to_ids to convert sentences to character ids
    sentences = [['A', 'person', 'is', 'walking', 'forwards', 'and', 'waving', 'his', 'hand'],
                 ['A', 'human', 'is', 'walking', 'in', 'a', 'circle'],
                 ['A', 'person', 'is', 'playing', 'violin', 'while', 'singing']]

    # print(character_ids)

    embeddings, mask = to_get_bert_embeddings.get_vectors(sentences)

    # print(embeddings)
    print(embeddings.size())
