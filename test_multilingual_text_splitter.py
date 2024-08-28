#pip install nltk langid jieba polyglot pyicu

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import langid
import jieba

# Download necessary NLTK data
nltk.download('punkt')

# Supported languages with fallback methods
supported_languages = {
    'ar_JO': 'arabic',
    'ca_ES': 'catalan',
    'cs_CZ': 'czech',
    'cy_GB': 'welsh',
    'da_DK': 'danish',
    'de_DE': 'german',
    'el_GR': 'greek',
    'en_GB': 'english',
    'en_US': 'english',
    'es_ES': 'spanish',
    'es_MX': 'spanish',
    'fi_FI': 'finnish',
    'fr_FR': 'french',
    'hu_HU': 'hungarian',
    'is_IS': 'icelandic',
    'it_IT': 'italian',
    'ka_GE': 'georgian',
    'kk_KZ': 'kazakh',
    'lb_LU': 'luxembourgish',
    'ne_NP': 'nepali',
    'nl_BE': 'dutch',
    'nl_NL': 'dutch',
    'no_NO': 'norwegian',
    'pl_PL': 'polish',
    'pt_BR': 'portuguese',
    'pt_PT': 'portuguese',
    'ro_RO': 'romanian',
    'ru_RU': 'russian',
    'sr_RS': 'serbian',
    'sv_SE': 'swedish',
    'sw_CD': 'swahili',
    'tr_TR': 'turkish',
    'uk_UA': 'ukrainian',
    'vi_VN': 'vietnamese',
    'zh_CN': 'chinese'
}

def split_text(text, lang_code=None, max_chunk_size=200):
    # Detect language if not provided
    if lang_code is None:
        lang_code, _ = langid.classify(text)
    
    if lang_code not in supported_languages:
        raise ValueError(f"Language code {lang_code} is not supported.")
    
    lang = supported_languages[lang_code]
    
    if lang == 'chinese':
        # Chinese tokenization using jieba
        words = jieba.lcut(text)
    elif lang in ['ka_GE', 'kk_KZ', 'ne_NP', 'lb_LU', 'sw_CD']:
        # For unsupported languages by NLTK, using basic word splitting
        words = text.split()
    else:
        # Standard NLTK processing
        sentences = sent_tokenize(text, language=lang)
        words = []
        for sentence in sentences:
            words.extend(word_tokenize(sentence, language=lang))
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Example usage
text = "Your input text here."
language_code = 'en_US'  # You can specify or set to None for auto-detection
chunks = split_text(text, lang_code=language_code)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")

