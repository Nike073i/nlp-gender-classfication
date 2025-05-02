import re

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
