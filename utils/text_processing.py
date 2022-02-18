import logging
import operator
import pickle
import re
import string
from itertools import product
from pathlib import Path

import numpy as np
import progressbar
from many_stop_words import get_stop_words
from nltk.corpus import stopwords, words
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

logger = logging.getLogger(__name__)

FILE_PATH = Path(__file__).resolve().parent
class UnicodeReplacer:
    displayUmlauts = True
    _translate = True

    # Escaped codes (from unicode)
    long_codes = {
    '\\xe2\\x80\\x90': '-',
    '\\xe2\\x80\\x91': '-',
    '\\xe2\\x80\\x92': '-',
    '\\xe2\\x80\\x93': '-',
    '\\xe2\\x80\\x94': '-',
    '\\xe2\\x80\\x94': '-',
    '\\xe2\\x80\\x98': "'",
    '\\xe2\\x80\\x9b': "'",
    '\\xe2\\x80\\x9c': '"',
    '\\xe2\\x80\\x9c': '"',
    '\\xe2\\x80\\x9d': '"',
    '\\xe2\\x80\\x9e': '"',
    '\\xe2\\x80\\x9f': '"',
    '\\xe2\\x80\\xa6': '..',
    '\\xe2\\x80\\xb2': "'",
    '\\xe2\\x80\\xb3': "'",
    '\\xe2\\x80\\xb4': "'",
    '\\xe2\\x80\\xb5': "'",
    '\\xe2\\x80\\xb6': "'",
    '\\xe2\\x80\\xb7': "'",
    '\\xe2\\x81\\xba': "+",
    '\\xe2\\x81\\xbb': "-",
    '\\xe2\\x81\\xbc': "=",
    '\\xe2\\x81\\xbd': "(",
    '\\xe2\\x81\\xbe': ")",
    }
    codes = {
    '//': '/', # Double /
    ' ': ' ', # Double spaces
    '\\n': ' ', # Line feed to space

    # German UTF8 codes
    '\\xef\\xbf\\xbd': chr(246),

    # Currencies
    '\\xe2\\x82\\xac': ' Euro ',

    # Special characters
    '\\x80\\x99': "'", # Single quote 
    '\\xc2\\xa1': '!', # Inverted exclamation
    '\\xc2\\xa2': 'c', # Cent sign
    '\\xc2\\xa3': '#', # Pound sign
    '\\xc2\\xa4': '$', # Currency sign
    '\\xc2\\xa5': 'Y', # Yen sign
    '\\xc2\\xa6': '|', # Broken bar
    '\\xc2\\xa7': '?', # Section sign
    '\\xc2\\xa8': ':', # Diaerisis
    '\\xc2\\xa9': '(C)', # Copyright
    '\\xc2\\xaa': '?', # Feminal ordinal
    '\\xc2\\xab': '<<', # Double left
    '\\xc2\\xac': '-', # Not sign
    '\\xc2\\xad': '', # Soft hyphen
    '\\xc2\\xae': '(R)', # Registered sign
    '\\xc2\\xaf': '-', # Macron
    '\\xc2\\xb0': 'o', # Degrees sign
    '\\xc2\\xb1': '+-', # Plus minus
    '\\xc2\\xb2': '2', # Superscript 2
    '\\xc2\\xb3': '3', # Superscript 3
    '\\xc2\\xb4': '', # Acute accent
    '\\xc2\\xb5': 'u', # Micro sign
    '\\xc2\\xb6': '', # Pilcrow
    '\\xc2\\xb7': '.', # Middle dot
    '\\xc2\\xb8': '', # Cedilla
    '\\xc2\\xb9': '1', # Superscript 1
    '\\xc2\\xba': '', # Masculine indicator
    '\\xc2\\xbb': '>>', # Double right
    '\\xc2\\xbc': '1/4', # 1/4 fraction
    '\\xc2\\xbd': '1/2', # 1/2 Fraction
    '\\xc2\\xbe': '3/4', # 3/4 Fraction
    '\\xc2\\xbf': '?', # Inverted ?

    # German unicode escape sequences
    '\\xc3\\x83': chr(223), # Sharp s es-zett
    '\\xc3\\x9f': chr(223), # Sharp s ?
    '\\xc3\\xa4': chr(228), # a umlaut
    '\\xc3\\xb6': chr(246), # o umlaut
    '\\xc3\\xbc': chr(252), # u umlaut
    '\\xc3\\x84': chr(196), # A umlaut
    '\\xc3\\x96': chr(214), # O umlaut
    '\\xc3\\x9c': chr(220), # U umlaut

    # Scandanavian unicode escape sequences
    '\\xc2\\x88': 'A', # aelig
    '\\xc2\\xb4': 'A', # aelig
    '\\xc3\\x85': 'Aa', # Aring
    '\\xc3\\x93': 'O', # O grave
    '\\xc3\\xa4': 'a', # a with double dot
    '\\xc3\\xa5': 'a', # aring
    '\\xc3\\x86': 'AE', # AElig
    '\\xc3\\x98': '0', # O crossed
    '\\xc3\\x99': 'U', # U grave
    '\\xc3\\xa6': 'ae', # aelig
    '\\xc3\\xb0': 'o', # o umlaut
    '\\xc3\\xb2': 'o', # o tilde
    '\\xc3\\xb3': 'o', # o reverse tilde
    '\\xc3\\xb4': 'o', # Capital O circumflex
    '\\xc3\\xb8': 'o', # oslash

    # French (Latin) unicode escape sequences
    '\\xc3\\x80': 'A', # A grave
    '\\xc3\\x81': 'A', # A acute
    '\\xc3\\x82': 'A', # A circumflex
    '\\xc3\\x83': 'A', # A tilde
    '\\xc3\\x88': 'E', # E grave
    '\\xc3\\x89': 'E', # E acute
    '\\xc3\\x8a': 'E', # E circumflex
    '\\xc3\\xa0': 'a', # a grave
    '\\xc3\\xa1': 'a', # a acute
    '\\xc3\\xa2': 'a', # a circumflex
    '\\xc3\\xa7': 'c', # c cedilla
    '\\xc3\\xa8': 'e', # e grave
    '\\xc3\\xa9': 'e', # e acute
    '\\xc3\\xaa': 'e', # e circumflex
    '\\xc3\\xab': 'e', # e diaeresis
    '\\xc3\\xae': 'i', # i circumflex
    '\\xc3\\xaf': 'i', # i diaeresis
    '\\xc3\\xb7': "/", # Division sign
    '\\xc5\\x93': 'oe', # oe joined

    # Hungarian lower case
    '\\xc3\\xb3': 'o', # o circumflex 
    '\\xc3\\xad': 'i', # i accent
    '\\xc3\\xb5': 'o', # o tilde
    '\\xc5\\x91': 'o', # o 
    '\\xc5\\xb1': chr(252), # 
    '\\xc3\\xba': 'u', # u acute

    # Polish unicode escape sequences
    '\\xc4\\x84': 'A', # A,
    '\\xc4\\x85': 'a', # a,
    '\\xc4\\x86': 'C', # C'
    '\\xc4\\x87': 'c', # c'
    '\\xc4\\x98': 'E', # E,
    '\\xc4\\x99': 'e', # e,
    '\\xc5\\x81': 'L', # L/
    '\\xc5\\x82': 'l', # l/
    '\\xc5\\x83': 'N', # N'
    '\\xc5\\x84': 'n', # n'
    '\\xc5\\x9a': 'S', # S'
    '\\xc5\\x9b': 's', # s'
    '\\xc5\\xb9': 'Z', # Z'
    '\\xc5\\xba': 'z', # z'
    '\\xc5\\xbb': 'Z', # Z.
    '\\xc5\\xbc': 'z', # z.

    # Greek upper case
    '\\xce\\x91': 'A', # Alpha
    '\\xce\\x92': 'B', # Beta
    '\\xce\\x93': 'G', # Gamma
    '\\xce\\x94': 'D', # Delta
    '\\xce\\x95': 'E', # Epsilon
    '\\xce\\x96': 'Z', # Zeta
    '\\xce\\x97': 'H', # Eta
    '\\xce\\x98': 'TH', # Theta
    '\\xce\\x99': 'I', # Iota
    '\\xce\\x9a': 'K', # Kappa
    '\\xce\\x9b': 'L', # Lamda
    '\\xce\\x9c': 'M', # Mu
    '\\xce\\x9e': 'N', # Nu
    '\\xce\\x9f': 'O', # Omicron
    '\\xce\\xa0': 'Pi', # Pi
    '\\xce ': 'Pi', # Pi ?
    '\\xce\\xa1': 'R', # Rho
    '\\xce\\xa3': 'S', # Sigma
    '\\xce\\xa4': 'T', # Tau
    '\\xce\\xa5': 'Y', # Upsilon
    '\\xce\\xa6': 'F', # Fi
    '\\xce\\xa7': 'X', # Chi
    '\\xce\\xa8': 'PS', # Psi
    '\\xce\\xa9': 'O', # Omega

    # Greek lower case
    '\\xce\\xb1': 'a', # Alpha
    '\\xce\\xb2': 'b', # Beta
    '\\xce\\xb3': 'c', # Gamma
    '\\xce\\xb4': 'd', # Delta
    '\\xce\\xb5': 'e', # Epsilon
    '\\xce\\xb6': 'z', # Zeta
    '\\xce\\xb7': 'h', # Eta
    '\\xce\\xb8': 'th', # Theta
    '\\xce\\xb9': 'i', # Iota
    '\\xce\\xba': 'k', # Kappa
    '\\xce\\xbb': 'l', # Lamda
    '\\xce\\xbc': 'm', # Mu
    '\\xce\\xbd': 'v', # Nu
    '\\xce\\xbe': 'ks', # Xi
    '\\xce\\xbf': 'o', # Omicron
    '\\xce\\xc0': 'p', # Pi
    '\\xce\\xc1': 'r', # Rho
    '\\xce\\xc3': 's', # Sigma
    '\\xce\\xc4': 't', # Tau
    '\\xce\\xc5': 'y', # Upsilon
    '\\xce\\xc6': 'f', # Fi
    '\\xce\\xc7': 'x', # Chi
    '\\xce\\xc8': 'ps', # Psi
    '\\xce\\xc9': 'o', # Omega

    # Icelandic
    '\\xc3\\xbe': 'p', # Like a p with up stroke
    '\\xc3\\xbd': 'y', # y diaeresis

    # Italian characters
    '\\xc3\\xac': 'i', # i reverse circumflex
    '\\xc3\\xb9': 'u', # u reverse circumflex

    # Polish (not previously covered)
    '\\xc3\\xa3': 'a', # a tilde

    # Romanian
    '\\xc4\\x83': 'a', # a circumflex variant
    '\\xc3\\xa2': 'a', # a circumflex 
    '\\xc3\\xae': 'i', # i circumflex 
    '\\xc5\\x9f': 's', # s cedilla ?
    '\\xc5\\xa3': 's', # t cedilla ?
    '\\xc8\\x99': 's', # s with down stroke
    '\\xc8\\x9b': 't', # t with down stroke

    # Spanish not covered above
    '\\xc3\\xb1': 'n', # n tilde

    # Turkish not covered above
    '\\xc3\\xbb': 'u', # u circumflex
    '\\xc4\\x9f': 'g', # g tilde
    '\\xc4\\xb1': 'i', # Looks like an i
    '\\xc4\\xb0': 'I', # Looks like an I
    '\xe2\x80\x99': "'",
    '\xc3\xa9': 'e',
    }

    # UTF8 codes (Must be checked after above codes checked)
    short_codes = {
    '\\xa0': ' ', # Line feed to space
    '\\xa3': '#', # Pound character
    '\\xb4': "'", # Apostrophe 
    '\\xc0': 'A', # A 
    '\\xc1': 'A', # A 
    '\\xc2': 'A', # A 
    '\\xc3': 'A', # A 
    '\\xc4': 'A', # A 
    '\\xc5': 'A', # A 
    '\\xc6': 'Ae', # AE
    '\\xc7': 'C', # C 
    '\\xc8': 'E', # E 
    '\\xc9': 'E', # E 
    '\\xca': 'E', # E 
    '\\xcb': 'E', # E 
    '\\xcc': 'I', # I 
    '\\xcd': 'I', # I 
    '\\xce': 'I', # I 
    '\\xcf': 'I', # I 
    '\\xd0': 'D', # D
    '\\xd1': 'N', # N 
    '\\xd2': 'O', # O 
    '\\xd3': 'O', # O 
    '\\xd4': 'O', # O 
    '\\xd5': 'O', # O 
    '\\xd6': 'O', # O 
    '\\xd7': 'x', # Multiply
    '\\xd8': '0', # O crossed 
    '\\xd9': 'U', # U 
    '\\xda': 'U', # U 
    '\\xdb': 'U', # U 
    '\\xdc': 'U', # U umlaut
    '\\xdd': 'Y', # Y
    '\\xdf': 'S', # Sharp s es-zett
    '\\xe0': 'e', # Small a reverse acute
    '\\xe1': 'a', # Small a acute
    '\\xe2': 'a', # Small a circumflex
    '\\xe3': 'a', # Small a tilde
    '\\xe4': 'a', # Small a diaeresis
    '\\xe5': 'aa', # Small a ring above
    '\\xe6': 'ae', # Joined ae
    '\\xe7': 'c', # Small c Cedilla
    '\\xe8': 'e', # Small e grave
    '\\xe9': 'e', # Small e acute
    '\\xea': 'e', # Small e circumflex
    '\\xeb': 'e', # Small e diarisis
    '\\xed': 'i', # Small i acute
    '\\xee': 'i', # Small i circumflex
    '\\xf1': 'n', # Small n tilde
    '\\xf3': 'o', # Small o acute
    '\\xf4': 'o', # Small o circumflex
    '\\xf6': 'o', # o umlaut
    '\\xf7': '/', # Division sign
    '\\xf8': 'oe', # Small o strike through 
    '\\xf9': 'u', # Small u circumflex
    '\\xfa': 'u', # Small u acute
    '\\xfb': 'u', # u circumflex
    '\\xfd': 'y', # y circumflex
    '\\xc0': 'A', # Small A grave
    '\\xc1': 'A', # Capital A acute
    '\\xc7': 'C', # Capital C Cedilla
    '\\xc9': 'E', # Capital E acute
    '\\xcd': 'I', # Capital I acute
    '\\xd3': 'O', # Capital O acute
    '\\xda': 'U', # Capital U acute
    '\\xfc': 'u', # u umlaut
    '\\xbf': '?', # Spanish Punctuation
    '\\xb0': 'o', # Degrees symbol
    }

    # HTML codes (RSS feeds)
    HtmlCodes = {
    # Currency
    chr(156): '#', # Pound by hash
    chr(169): '(c)', # Copyright

    # Norwegian
    chr(216): '0', # Oslash

    # Spanish french
    chr(241): 'n', # Small tilde n
    chr(191): '?', # Small u acute to u
    chr(224): 'a', # Small a grave to a
    chr(225): 'a', # Small a acute to a
    chr(226): 'a', # Small a circumflex to a
    chr(232): 'e', # Small e grave to e
    chr(233): 'e', # Small e acute to e
    chr(234): 'e', # Small e circumflex to e
    chr(235): 'e', # Small e diarisis to e
    chr(237): 'i', # Small i acute to i
    chr(238): 'i', # Small i circumflex to i
    chr(243): 'o', # Small o acute to o
    chr(244): 'o', # Small o circumflex to o
    chr(250): 'u', # Small u acute to u
    chr(251): 'u', # Small u circumflex to u
    chr(192): 'A', # Capital A grave to A
    chr(193): 'A', # Capital A acute to A
    chr(201): 'E', # Capital E acute to E
    chr(205): 'I', # Capital I acute to I
    chr(209): 'N', # Capital N acute to N
    chr(211): 'O', # Capital O acute to O
    chr(218): 'U', # Capital U acute to U
    chr(220): 'U', # Capital U umlaut to U
    chr(231): 'c', # Small c Cedilla
    chr(199): 'C', # Capital C Cedilla

    # German
    chr(196): "Ae", # A umlaut
    chr(214): "Oe", # O umlaut
    chr(220): "Ue", # U umlaut
    }

    unicodes = {
    '\\u201e': '"', # ORF feed
    '\\u3000': " ",
    '\\u201c': '"',
    '\\u201d': '"',
    '\\u0153': "oe", # French oe
    '\\u2009': ' ', # Short space to space
    '\\u2013': '-', # Long dash to minus sign
    '\\u2018': "'", # Left single quote
    '\\u2019': "'", # Right single quote

    # Czech
    '\\u010c': "C", # C cyrillic
    '\\u010d': "c", # c cyrillic
    '\\u010e': "D", # D cyrillic
    '\\u010f': "d", # d cyrillic
    '\\u011a': "E", # E cyrillic
    '\\u011b': "e", # e cyrillic
    '\\u013a': "I", # I cyrillic
    '\\u013d': "D", # D cyrillic
    '\\u013e': "I", # I cyrillic
    '\\u0139': "L", # L cyrillic
    '\\u0147': "N", # N cyrillic
    '\\u0148': "n", # n cyrillic
    '\\u0154': "R", # R cyrillic
    '\\u0155': "r", # r cyrillic
    '\\u0158': "R", # R cyrillic
    '\\u0159': "r", # r cyrillic
    '\\u0160': "S", # S cyrillic
    '\\u0161': "s", # s cyrillic
    '\\u0164': "T", # T cyrillic
    '\\u0165': "t", # t cyrillic
    '\\u016e': "U", # U cyrillic
    '\\u016f': "u", # u cyrillic
    '\\u017d': "Z", # Z cyrillic
    '\\u017e': "z", # z cyrillic
    }

    # Convert escaped characters (umlauts etc.) to normal characters
    def replace(self, text):
        s = text
        for code in self.long_codes.keys():
            s = s.replace(code, self.long_codes[code])

        for code in self.codes.keys():
            s = s.replace(code, self.codes[code])

        for code in self.short_codes.keys():
            s = s.replace(code, self.short_codes[code])

        s = s.replace("'oC", 'oC') # Degrees C fudge
        s = s.replace("'oF", 'oF') # Degrees C fudge
        return s

def get_twitter_stop_words():
    return [
    'rt', 'the', 'ora', 'mai', 'brt', 'via', 'poi', 'cos', 'retweeted',
    'youtube', 'retweeted', 'and', 'new', 'italia', 'solo', 'xac', 'xec',
    'oggi', 'quando', 'oggi', 'grazie', 'anni', 'fatto', 'video',
    'sempre', 'sempre', 'cosa', 'essere', 'bene', 'dopo',
    'deve', 'grande', 'xcxa', 'dire', 'parte', 'così', 'buona',
    'buongiorno', 'della', 'delle', 'devo', 'devi', 'pixcxb', 'perchxcxa'
    ]

def get_italian_stop_words():
    stop_words= ['a',
     'adesso',
     'ai',
     'al',
     'alla',
     'allo',
     'allora',
     'almeno',
     'altre',
     'altri',
     'altro',
     'anche',
     'ancora',
     'avere',
     'aveva',
     'avevano',
     'ben',
     'buono',
     'che',
     'chi',
     'cinque',
     'comprare',
     'con',
     'consecutivi',
     'consecutivo',
     'cosa',
     'cui',
     'da',
     'dà',
     'dá',
     'del',
     'della',
     'dello',
     'dentro',
     'deve',
     'devo',
     'di',
     'doppio',
     'due',
     'e',
     'ecco',
     'fare',
     'fine',
     'fino',
     'fra',
     'gente',
     'giu',
     'ha',
     'hai',
     'hanno',
     'ho',
     'il',
     'indietro',
     'invece',
     'io',
     'la',
     'lavoro',
     'le',
     'lei',
     'lo',
     'loro',
     'lui',
     'lungo',
     'ma',
     'me',
     'meglio',
     'molta',
     'molti',
     'molto',
     'neanche',
     'nei',
     'nella',
     'niente',
     'no',
     'noi',
     'nome',
     'nostro',
     'nove',
     'nuovi',
     'nuovo',
     'o',
     'oltre',
     'ora',
     'otto',
     'peggio',
     'perche',
     'perchè',
     'perché',
     'pero',
     'però',
     'peró',
     'persone',
     'piu',
     'più',
     'piú',
     'poco',
     'primo',
     'promesso',
     'proprio',
     'qua',
     'quarto',
     'quasi',
     'quattro',
     'quel',
     'quello',
     'questo',
     'qui',
     'quindi',
     'quinto',
     'rispetto',
     'sara',
     'secondo',
     'sei',
     'sembra',
     'sembrava',
     'senza',
     'sette',
     'sia',
     'siamo',
     'siete',
     'solo',
     'sono',
     'sopra',
     'soprattutto',
     'sotto',
     'stati',
     'stato',
     'stesso',
     'su',
     'subito',
     'sul',
     'sulla',
     'tanto',
     'te',
     'tempo',
     'terzo',
     'tra',
     'tre',
     'triplo',
     'ultimo',
     'un',
     'una',
     'uno',
     'va',
     'vai',
     'voi',
     'volte',
     'vostro']
    return stop_words

def not_delete_words():
    return ["pd", "m5s", "fi", "fdi"]
   

def lowercase(d):
    return d.lower()


def apostrophes(s):
    return s.replace("'", "")


def letter_digit_underscore(d):
    pat = r'[^\wáéíóúàèìòùÁÉÍÓÚÀÈÌÒÙ]'
    return re.sub(pat, ' ', d)


def remove_space_newline_tab(d):
    return re.sub(r'\s', ' ', d)


def remove_url(d):
    return re.sub(r'http\S+', '', d)


table = str.maketrans(dict.fromkeys(string.punctuation))


def remove_punctuation(d):
    return d.translate(table)


def remove_digits(d):
    return re.sub('\d+', '', d)

SENTENCE_PROCESS = [remove_url, remove_punctuation]

DEFAULT_PIPELINE = [
    lowercase, apostrophes, letter_digit_underscore,
    remove_space_newline_tab, remove_digits
]


def min_length(d, min_l=4):
    return len(d) >= min_l

with open(FILE_PATH / 'google-10000-english.txt', 'r') as file:
    englsh_words_set = set(file.read().split('\n'))   

def is_not_english(d):
    return (d not in englsh_words_set)


DEFAULT_VALIDATION = [min_length, is_not_english]


def validate_word(validation, word):
    valid = True
    for validation_fun in validation:
        if not validation_fun(word):
            valid = False
            break
    return valid


def clean_sentence(sentence,
                   process_sentence=SENTENCE_PROCESS,
                   pipeline=DEFAULT_PIPELINE,
                   stop_words=[],
                   validation=DEFAULT_VALIDATION,
                   join=False,
                   tokenizer=word_tokenize,
                   keep_words=not_delete_words()):

    try:
        sentence = bytes(sentence, encoding='latin1').decode('utf-8')   
    except:
        pass
    sentence = UnicodeReplacer().replace(sentence)
    for process_fun in process_sentence:
        sentence = process_fun(sentence)
    final_sentence = []
    for word in tokenizer(sentence):
        orig_word = word
        if word in keep_words:
            final_sentence.append(word)
            continue
        for process_fun in pipeline:
            word = process_fun(word)
        if word in stop_words:
            continue
        if not validate_word(validation, word):
            continue
        final_sentence.append(word)
    if len(final_sentence) == 0:
        return ""
    if join:
        return ' '.join(final_sentence)
    else:
        return final_sentence


def vocabulary_size(data):
    logging.info('Computing Vocabulary size')
    d = set()
    for sentence in data:
        d.update(w for w in word_tokenize(sentence))
    return len(d)


def clean_content(data,
                  pipeline=DEFAULT_PIPELINE,
                  stop_words=[],
                  validation=DEFAULT_VALIDATION,
                  join=False,
                  tokenizer=word_tokenize):
    """Clean text data

    Arguments:
        data: list of str
              The list of documents to be cleaned

    Keyword Arguments:
        pipeline: list of functions
                   A sequence of function to be applied to every document
                   default=DEFAULT_PIPELINE
        stop_words: iterable
                    Stop words to remove from each document
        join: bool
              Wheter to join the tokenized document or not

    Returns:
        list of str
        Document processed.
    """
    logger.info('Cleaning content')
    documents = []
    for d in progressbar.progressbar(data):
        documents.append(
            clean_sentence(d, pipeline, stop_words, validation, join,
                           tokenizer))
    return documents


def get_stopwords(lang):
    lang_mapping = {
        'en': 'english',
        'es': 'spanish',
        'pt': 'portuguese',
        'it': 'italian'
    }
    stop_words = set(get_stop_words(lang))
    nltk_words = set(stopwords.words(lang_mapping[lang]))
    m_stop_words = set(get_stop_words(lang))
    stop_words.update(nltk_words)
    stop_words.update(m_stop_words)
    stop_words.update(get_twitter_stop_words())
    stop_words.update(get_italian_stop_words())
    return stop_words


def get_topics_distribution(categories, min_occurences=12):
    """
    Get Topics Distribution

    Parameters
    ----------
    categories: dict
                A dictionary where the keys are the topics, and the values are the words associated with each topic.
    """
    probabilities = dict()

    for category in categories.keys():
        probabilities[category] = dict()
        for w in categories[category]:
            if w not in probabilities[category]:
                probabilities[category][w] = 0
            probabilities[category][w] = probabilities[category][w] + 1
        remove = []
        for w in probabilities[category].keys():
            if probabilities[category][w] < min_occurences:
                remove.append(w)
        for k in remove:
            del probabilities[category][k]
        cant_words = sum(probabilities[category].values())
        probabilities[category] = {
            k: v / cant_words
            for k, v in probabilities[category].items()
        }

    return probabilities


def get_p_b_d(document, wa, wb):
    """ Given a biterm obtain its probability in the docuemnt """
    n_biterms = 0.0
    biterm = 0.0
    for w1, w2 in product(document, document):
        n_biterms = n_biterms + 1
        if (w1 == wa) and (w2 == wb):
            biterm = biterm + 1
    return biterm / n_biterms


def topic_assignment(question, topic_distribution):
    total_words = sum([len(topic) for topic in topic_distribution.values()])
    p_z = {
        topic_name: len(topic) / total_words
        for topic_name, topic in topic_distribution.items()
    }
    p_z_d = dict()
    for j in topic_distribution.keys():
        p_z_d[j] = 0.0
    for w1, w2 in product(question, question):
        p_b_d = get_p_b_d(question, w1, w2)
        temp = dict()
        for j in topic_distribution.keys():
            t_d_w1 = 0.00000001
            t_d_w2 = 0.00000001
            if w1 in topic_distribution[
                    j] and topic_distribution[j][w1] > 0.00000001:
                t_d_w1 = topic_distribution[j][w1]
            if w2 in topic_distribution[
                    j] and topic_distribution[j][w2] > 0.00000001:
                t_d_w2 = topic_distribution[j][w2]
            temp[j] = p_z[j] * t_d_w1 * t_d_w2
        deno = sum(temp.values()) + 0.00000001
        p_z_b = {j: t / deno for j, t in temp.items()}
        for j in topic_distribution.keys():
            p_z_d[j] = p_z_d[j] + p_z_b[j] * p_b_d
    if all(value <= 0.00000001 * 0.00000001 for value in p_z_d.values()):
        prob = 'general'
    else:
        prob = max(p_z_d.items(), key=operator.itemgetter(1))[0]
    return prob
