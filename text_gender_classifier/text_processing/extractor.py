from collections import defaultdict, Counter
import json
import re
from itertools import chain
from .cleaner import sentence_processing_pipeline
from .config import regex_dict, pos_tags_list, punct_list, token_dict
from .nlp_utils import break_into_sentences, pos_analyze, stopwords_ru


def process_text(text):
    sentences = break_into_sentences(text)
    preparing_sentences = [ sentence_processing_pipeline.execute(sentence.text) for sentence in sentences ]
   
    return preparing_sentences

def analyze_sentence(sentence):
    without_tags = regex_dict['special_token'].sub('', sentence)
    return pos_analyze(without_tags)

def count_punctuation_and_tokens(sentence):
    counter = defaultdict(int)
    for elem in re.finditer(regex_dict['non_words'], sentence):
        counter[elem.group()] += 1    
    return counter

def count_pos_tag_usage(parts):
    tags = [ part[1] for part in parts ]
    return Counter(tags)

def get_mean_token_length(sentences):
    words = list(chain.from_iterable(sentences))
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def extract_features(text):
    sentences = process_text(text)
    sentences_count = len(sentences)    

    lemm_sentences = []

    x = defaultdict(int, {
        **{token: 0 for token in token_dict.values()}, 
        **{punct: 0 for punct in punct_list } },
        **{tag: 0 for tag in pos_tags_list }
    )

    for sentence in sentences:
        parts = analyze_sentence(sentence)
        word_parts = [ part for part in parts if part[2] not in stopwords_ru and part[1] != 'PUNCT' and part[1] != 'SYM' and part[1] != 'X' ]
        symbols_usage = count_punctuation_and_tokens(sentence)
        parts_usage = count_pos_tag_usage(word_parts)

        for key, value in symbols_usage.items(): 
            x[key] += value

        for key, value in parts_usage.items(): 
            x[key] += value

        x['sentence_length'] += len(parts)

        lemm_sentence = [ part[2] for part in word_parts ]
        if len(lemm_sentence) > 0:
            lemm_sentences.append(lemm_sentence)

    text_features = { f"mean_usage_{key}": value / sentences_count for key, value in x.items() }
    text_features['mean_token_length'] = get_mean_token_length(lemm_sentences)
    text_features['lemm_sentences'] = json.dumps(lemm_sentences, ensure_ascii=False, indent=None)
    text_features['sentences_count'] = sentences_count

    return text_features
