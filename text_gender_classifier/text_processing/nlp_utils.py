from urlextract import URLExtract
from natasha import (
    Segmenter, MorphVocab, Doc,
    NewsEmbedding, NewsMorphTagger,
    AddrExtractor, DatesExtractor, MoneyExtractor)
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import phonenumbers
from bs4 import BeautifulSoup
from razdel import sentenize

nltk.download("stopwords")
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
url_extractor = URLExtract()

stopwords_ru = set(stopwords.words("russian"))

def extract_urls(text):
    return url_extractor.find_urls(text)

def extract_addr(text):
    return addr_extractor(text)

def extract_date(text):
    return dates_extractor(text)

def extract_money(text):
    return money_extractor(text)

spell_ru = SpellChecker(language='ru')
def correct_spelling(word):
    return spell_ru.correction(word)

def is_exists_word(word):
    return word in spell_ru

def get_phone_matches(text):
    return phonenumbers.PhoneNumberMatcher(text, region='ru')

def extract_html_text(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def break_into_sentences(text):
    return sentenize(text)

def pos_analyze(text):
    doc = Doc(text)
    doc.segment(segmenter) 
    doc.tag_morph(morph_tagger)
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    return [ (token.text, token.pos, token.lemma) for token in doc.tokens ]