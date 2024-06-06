from json import load
from typing import Any

from random import random
import torch
from navec import Navec
import spacy
from torch import Tensor

from core.scripts.config_manager import get_config

config = get_config()['multiclass_classifier']

SPC_SYMBOLS = set(r"""!"#$%&'()*+-–/:;<=>?@[\]^_`{|}~«»""")
spc_dict = {ord(char): None for char in SPC_SYMBOLS}
spc_dict.update({ord('.'): ord(' '), ord(','): ord(' ')})
clear = lambda x: x.translate(spc_dict)

Navec_model = Navec.load(config['word2vec_path'])
sp = spacy.load("ru_core_news_md")
UNK = torch.rand((1, 300), dtype=torch.float32)
EMP = torch.rand((1, 300), dtype=torch.float32)
wrong_size = torch.Size((1, 300))


def tokenize(texts: list) -> list[Tensor | Any]:
    if type(texts) not in (list, tuple):
        texts = [texts]
    result = []
    for text in texts:
        samples = []
        for word in sp(text):
            try:
                temp = Navec_model[word.lemma_]
            except KeyError:
                samples.append(UNK)
            else:
                samples.append(torch.tensor(temp, dtype=torch.float32).reshape([1, 300]))
        if len(samples) == 0:
            result.append(EMP)

        result.append(torch.mean(torch.stack(samples), 0) if len(samples) > 1 else samples[0])

    return result


def load_config(path: str) -> tuple:
    with open(path, 'r', encoding='utf-8') as f:
        config = load(f)
    return config['label_num'], config['samples'], config['name'], config['augmentation']


def augment_binary(samples: list[str], augment: list[str], chance: int) -> list:
    new = []
    chance = chance / 100.0
    for s in samples:
        for a in augment:
            new.append((a + " " + s) if random() < chance else s)
    return new


class OutOfRangeException(Exception):
    def __init__(self, label: int, first_num: int , max_num: int):
        self.message = f"Label {label} out of range: [{first_num}, {max_num}]"
        super().__init__(self.message)


class EmptyTextException(Exception):
    def __init__(self):
        self.message = "Argument is empty"
        super().__init__(self.message)


class OneHot:
    def __init__(self, labels_num: int, first_num: int = 0, dtype: torch.dtype = torch.long):
        self.labels_num = labels_num
        self.first_num = first_num
        self.dtype = dtype
        self.max_num = self.first_num + self.labels_num - 1
    def __call__(self, label: int) -> torch.Tensor:
        if not self.first_num <= label <= self.max_num:
            raise OutOfRangeException(label, self.first_num, self.max_num)
        label_one_hot = torch.zeros(self.labels_num, dtype=self.dtype)
        label_one_hot[label] += 1
        return label_one_hot


def load_filters() -> list:
    with open(get_config()['multiclass_classifier']['filter_path'], 'r', encoding='utf-8') as f:
        x = [(i[:-1] if i.endswith('\n') else i) for i in set(f.readlines())]
        return x


class Filter:
    def __init__(self, filters: list | tuple = None, use_spacy: bool = False):
        self.filters = filters if filters is not None else load_filters()
        self.use_spacy = use_spacy

    def update_filters(self):
        self.filters = load_filters()

    def __call__(self, text: str) -> str:
        if len(text) == 0:
            raise EmptyTextException
        for f in self.filters:
            text = text.replace(f, "").strip()
        text = " ".join(text.split())
        if self.use_spacy:
            text = " ".join([i.lemma_ for i in sp(text)])
        return text
