from .nlp_utils import (
    extract_urls, extract_addr, extract_date, extract_money, 
    correct_spelling, is_exists_word, get_phone_matches, extract_html_text
)
from .config import regex_dict, token_dict

def clean_html(text):
    if not text: return ""
    text = extract_html_text(text)
    text = regex_dict['html_shielding'].sub("", text)
    return text

def replace_e(text):
    return text.replace('ё', 'е')

def replace_urls(text):
    urls = extract_urls(text)
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
    for match in get_phone_matches(text):
        text = text.replace(match.raw_string, token_dict['phone'])
    return text

def replace_addresses(text):
    text = regex_dict['address'].sub(token_dict['address'], text)
    text = replace_with_extractor(extract_addr, text, token_dict['address'])
    return text

def replace_dates(text):
    text = regex_dict['date'].sub(token_dict['date'], text)
    text = regex_dict['time'].sub(token_dict['time'], text)
    text = replace_with_extractor(extract_date, text, token_dict['date'])
    return text

def replace_quantities(text):
    text = replace_with_extractor(extract_money, text, token_dict['quantity'])
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
        if word.isalpha() and not is_exists_word(word):
            corrected = correct_spelling(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

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

sentence_processing_pipeline = (
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