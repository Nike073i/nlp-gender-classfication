import os
import math
from bs4 import BeautifulSoup
import re
import pandas as pd
from razdel import sentenize
from collections import defaultdict, Counter
import json
from urlextract import URLExtract
import phonenumbers
from natasha import (
    Segmenter, MorphVocab, Doc,
    NewsEmbedding, NewsMorphTagger,
    AddrExtractor, DatesExtractor, MoneyExtractor)
import nltk
from nltk.corpus import stopwords
from itertools import chain
from spellchecker import SpellChecker
from concurrent.futures import ProcessPoolExecutor, as_completed

nltk.download("stopwords")
spell_ru = SpellChecker(language='ru')

stopwords_ru = set(stopwords.words("russian"))
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
addr_extractor = AddrExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
url_extractor = URLExtract()

token_dict =  {
    "url": "[URL]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "address": "[ADDRESS]",
    "date": "[DATE]",
    "time": "[TIME]",
    "quantity": "[QUANTITY]",
    "ordinary": "[ORDINARY]",
    "number": "[NUMBER]",
    "quoute": "[QUOTE]",
    "smile": "[SMILE]",
    "punctiuation_emotion": "[PUNCEM]",
    'removed': "[REMOVED]"
}
punct_list = [ ',', '!', '?', ';', ':', '-', '(', ')' ]
pos_tags_list = [ 'ADJ', 'ADP', 'ADV', 'AUX',  'INTJ', 'CCONJ', 'NOUN', 'DET',  'PROPN', 'NUM', 'VERB', 'PART',  'PRON',  'SCONJ' ]

regex_dict = {
    'html_shielding': re.compile(r"&\w+;"),  # экраны - &nbsp;
    "email": re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'),
    "address": re.compile(
        r'\b(ул\.|улица|просп\.|проспект|б[- ]?р\.|бульвар|пл\.|площадь|'
        r'пер\.|переулок|проезд)\s*([а-яА-Я-]+(\s*[а-яА-Я-]+)*)'
        r'(\s*[,\.]?\s*(д\.?|дом)?\s*\d+\s*[а-яА-Я]?)?'
        r'(\s*[,\.]?\s*(кв\.|квартира)?\s*\d+\s*[а-яА-Я]?)?'
    ),
    "date": re.compile(
        r'\b((0?[1-9]|[12][0-9]|3[01])[./-](0?[1-9]|1[0-2])[./-](19|20)?\d{2}|'
        r'(19|20)\d{2}[./-](0?[1-9]|1[0-2])[./-](0?[1-9]|[12][0-9]|3[01])|'
        r'(0?[1-9]|1[0-2])[./-](0?[1-9]|[12][0-9]|3[01])[./-](19|20)?\d{2})\b'
    ),
    "time": re.compile(r'\b(0?[0-9]|1[0-9]|2[0-3])[:ч.,\s-][0-5][0-9]\b'),
    "quantity_units": re.compile(
        r'(\d+[\d.,]*)\s*(кг|кило|шт|штук|ед|г|м|см|км|л|сек|мин|ч|лет|год)\b'
    ),
    "ordinal_prefix": re.compile(r'(пункт|п\.|раздел|гл|глава|статья|ст|№)\s*\d+[\d.-]*'),
    "ordinal_dot": re.compile(r'\b(\d+)\.(?=\s)'),
    "number": re.compile(r'\b[\+-]?\d+([.,]\d+)?\b'),
    "quotes": re.compile(r'«[^»]*»|“[^”]*”|"[^"]*"'),
    
    "smileys": re.compile(
        r'('
        r'[:=;][-^~*]?[)dp(]+|'      # :) :-D ;P =(
        r'[)dp(]+[-^~*]?[:=;]|'      # D: )))): 
        r'[0o]_[0o]|'                # 0_0 o_O
        r'[><t^][-_^][><t^]|'        # >_< ^_^ T_T
        r'x[dpo]+|'                  # xd xD
        r'[-\\/][_:][-\\/]|'         # -/- :-/
        r'[\[\]][-_][\[\]]|'         # :-[ ]-:
        r'\b((хи)+х?|(хе)+х?|(ха)+х?)\b'   # хихи хехе хахах
        r')'
    ),
    "emotion_punct": re.compile(r'[.!?]{2,}'),
    "non_symbol_chars": re.compile(r'([^\w\s\-])'), 
    "special_token_name": re.compile(r'\[\s*([^\_]+?)\s*\]'), 
    "special_token": re.compile(r'(\[[A-Z]+\])'),
    "multi_spaces": re.compile(r'\s+'),
    "english": re.compile(r'\b[a-z]+(-[a-zA-Z]+)*\b'), # токены не затронет, поскольку они в UpperCase
    "non_words": re.compile(r'(\[[A-Z]+\])|[,!?;:\-#()]')
}

def clean_html(text):
    if not text: return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = regex_dict['html_shielding'].sub("", text)
    return text

def replace_e(text):
    return text.replace('ё', 'е')

def replace_urls(text):
    urls = url_extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, token_dict['url'])
    return text

def replace_with_extractor(extractor, text, token):
    matches = extractor(text)
    spans = []
    for match in matches:
        spans.append((match.start, match.stop))

    sortedSpans = sorted(spans, key=lambda x: x[0], reverse=True)
    for span in sortedSpans:
        text = text[:span[0]] + token + text[span[1]:]
    return text

def replace_emails(text):
    return regex_dict['email'].sub(token_dict['email'], text)

def replace_phones(text):
    for match in phonenumbers.PhoneNumberMatcher(text, region='ru'):
        text = text.replace(match.raw_string, token_dict['phone'])
    return text

def replace_addresses(text):
    text = regex_dict['address'].sub(token_dict['address'], text)
    text = replace_with_extractor(addr_extractor, text, token_dict['address'])
    return text

def replace_dates(text):
    text = regex_dict['date'].sub(token_dict['date'], text)
    text = regex_dict['time'].sub(token_dict['time'], text)
    text = replace_with_extractor(dates_extractor, text, token_dict['date'])
    return text

def replace_quantities(text):
    text = replace_with_extractor(money_extractor, text, token_dict['quantity'])
    return regex_dict['quantity_units'].sub(token_dict['quantity'], text)

def replace_ordinals(text):
    text = regex_dict['ordinal_prefix'].sub(token_dict['ordinary'], text)
    text = regex_dict['ordinal_dot'].sub(token_dict['ordinary'], text)
    text = regex_dict['number'].sub(token_dict['number'], text)
    return text

def replace_quotes(text):
    return regex_dict['quotes'].sub(token_dict['quoute'], text)

def replace_emoticons(text):
    return regex_dict['smileys'].sub(token_dict['smile'], text)

def replace_punctuation_combinations(text):
    return regex_dict['emotion_punct'].sub(token_dict['punctiuation_emotion'], text)

def replace_english(text):
    return regex_dict['english'].sub(token_dict['removed'], text)
    

def split_special_chars(text):
    text = regex_dict['non_symbol_chars'].sub(r' \1 ', text)
    text = regex_dict['special_token_name'].sub(r'[\1]', text)
    return regex_dict['multi_spaces'].sub(' ', text).strip()

def to_lower(text):
    return text.lower()


def fix_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        # Исправляем только слова с ошибками (игнорируем числа, токены и т.д.)
        if word.isalpha() and word not in spell_ru:
            corrected = spell_ru.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def process_text(text):
    sentences = sentenize(text)
    preparing_sentences = [ sentence_processing.execute(sentence.text) for sentence in sentences ]
   
    return preparing_sentences


class Pipeline:
    def __init__(self, handler):
        self.handler = handler

    def execute(self, input):
        return self.handler(input)

    def then(self, handler):
        old_handler = self.handler
        def composed(x):
            output = old_handler(x)
            return handler(output)
        self.handler = composed
        return self 

sentence_processing = (
    Pipeline(clean_html)
    .then(to_lower)
    .then(replace_e)
    .then(replace_emails)
    .then(replace_urls)
    .then(replace_phones)
    .then(replace_addresses)
    .then(replace_dates)
    .then(replace_quantities)
    .then(replace_ordinals)
    .then(replace_quotes)
    .then(replace_emoticons)
    .then(replace_punctuation_combinations)
    .then(replace_english)
    .then(split_special_chars)
    .then(fix_spelling)
)

def get_parts(sentence):
    without_tags = regex_dict['special_token'].sub('', sentence)

    doc = Doc(without_tags)
    doc.segment(segmenter) 
    doc.tag_morph(morph_tagger)
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    
    return [ (token.text, token.pos, token.lemma) for token in doc.tokens ]

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
    print(f"Текст - {text[:10]}")

    sentences = process_text(text)
    sentences_count = len(sentences)    

    lemm_sentences = []

    x = defaultdict(int, {
        **{token: 0 for token in token_dict.values()}, 
        **{punct: 0 for punct in punct_list } },
        **{tag: 0 for tag in pos_tags_list }
    )

    for sentence in sentences:
        parts = get_parts(sentence)
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

def get_start_chunk(chunk_dir):
    existing_chunks = []
    for f in os.listdir(chunk_dir):
        if f.startswith("chunk_") and f.endswith(".parquet"):
            try:
                num = int(f.split('_')[1].split('.')[0])
                existing_chunks.append(num)
            except ValueError:
                continue
    
    return max(existing_chunks) + 1 if existing_chunks else 0

def combine_chunks(chunk_dir, output_file):
    chunk_files = sorted(
        [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) 
         if f.startswith("chunk_") and f.endswith(".parquet")],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    final_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    final_df.to_excel(output_file, index=False)

def process_single_chunk(args):
    chunk_num, input_file, chunk_size, header, text_column, process_func, temp_dir = args
    try:
        skiprows = 1 + chunk_num * chunk_size
        
        chunk = pd.read_excel(
            input_file,
            skiprows=skiprows,
            nrows=chunk_size,
            header=None,
            names=header
        ).dropna()
        
        processed_data = chunk[text_column].apply(process_func)
        features_df = pd.json_normalize(processed_data.tolist())
        
        chunk = pd.concat(
            [chunk.reset_index(drop=True), 
             features_df.reset_index(drop=True)], 
            axis=1
        )
        
        chunk.to_parquet(os.path.join(temp_dir, f"chunk_{chunk_num}.parquet"))
        return True
    except Exception as e:
        print(f"Ошибка в чанке {chunk_num}: {str(e)}")
        return False

def chunk_text_processing(
    input_file,
    output_file,
    text_column,
    process_func,
    chunk_size=100,
    temp_dir="temp_chunks",
    max_workers=None
):
    os.makedirs(temp_dir, exist_ok=True)
    
    start_chunk = get_start_chunk(temp_dir)
    
    header = pd.read_excel(input_file, nrows=0).columns.tolist()
    total_rows = len(pd.read_excel(input_file)) - 1
    total_chunks = math.ceil(total_rows / chunk_size)
    
    if start_chunk < total_chunks:
        tasks = [
            (chunk_num, input_file, chunk_size, header, 
             text_column, process_func, temp_dir)
            for chunk_num in range(start_chunk, total_chunks)
        ]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_chunk, task) for task in tasks]
            
            for future in as_completed(futures):
                success = future.result()
                if not success:
                    print("Обнаружена ошибка при обработке чанка")
    
    combine_chunks(temp_dir, output_file)
    print(f"Готово! Результат сохранен в {output_file}")

chunk_text_processing(
    input_file="./data/raw.xlsx",
    output_file="./data/processed.xlsx",
    text_column="Текст",
    process_func=extract_features,
    chunk_size=500,
    max_workers=os.cpu_count()
)
