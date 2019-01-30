import re
import string
from typing import Sequence, Optional, Mapping, Tuple

import numpy as np
import tensorflow as tf

PUNCT = str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
SPACE = re.compile("\s+")
REP_PAT = re.compile(r"(.)\1{2,}", re.DOTALL)


def text2id_w_padding(texts: Sequence[str], vocab: Mapping[str, int],
                      unk_idx: int, max_len: int, is_lower: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of strings into index according to provided vocabulary.
    The list will be convert to Numpy type, post padded with 0 
    :param texts: list of strings
    :param vocab: dictionary of token to index
    :param unk_idx: UNK token index
    :param max_len: Max padding length
    :param is_lower: If lower case all tokens
    :return: tuple of padded index sequence, lengths (np.array)
    """
    sequence, lengths = text2id(texts, vocab, unk_idx, max_len, is_lower)
    return tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=max_len, truncating="post"),\
           np.array(lengths)


def text2id(texts: Sequence[str], vocab: Mapping[str, int], unk_idx: int,
            max_len: int, is_lower: Optional[bool] = False) -> Tuple[Sequence, Sequence]:
    """
    Convert list of strings into index according to provided vocabulary.
    :param texts: list of strings
    :param vocab: dictionary of token to index
    :param unk_idx: UNK token index
    :param max_len: Max padding length
    :param is_lower: If lower case all tokens
    :return: tuple of index sequence (list of lists), lengths (lists)
    """
    sequence = []
    lengths = []
    for text in texts:
        if is_lower:
            text = text.lower()
        toks = [vocab.get(tok, unk_idx) for tok in text.split(" ")]
        sequence.append(toks)
        lengths.append(len(toks) if len(toks) < max_len else max_len)

    return sequence, lengths


def add_space_to_punct(text: str) -> str:
    """
    Add white space to punctuation
    E.g.: Today is great! --> Today is great !
    :param text: 
    :return: 
    """
    return text.translate(PUNCT)


def remove_extra_space(text: str) -> str:
    """
    Remove any (extra) spaces (tab, newline, spaces) and replace with single white space
    :param text: 
    :return: 
    """
    return SPACE.sub(" ", text)


def remove_dup_chars(text: str) -> str:
    """
    Remove duplicated characters (if more than 2)
    E.g.: 
    Awwwwwww --> Aw
    Whatttttt --> What
    Loook --> Lok (can't really fix this one)
    Look --> Look
    :param text: 
    :return: 
    """
    return REP_PAT.sub(r"\1", text)
