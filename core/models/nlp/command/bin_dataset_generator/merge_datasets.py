from glob import glob
from json import dump, load
from random import sample as pick
from num2words import num2words

from core.scripts.config_manager import get_config
from core.scripts.tools import Response

SPC_SYMBOLS = set(r"""!"#$%&'()*+-–/:;<=>?@[\]^_`{|}~«»""")
spc_dict = {ord(char): None for char in SPC_SYMBOLS}
spc_dict.update({ord('.'): ord(' '), ord(','): ord(' ')})
clear = lambda x: x.translate(spc_dict)

config = get_config()['binary_classifier']


def get_data(split_len=False, samples_count=int(config['samples_count'])) -> list:
    datasets = glob(config['dataset_input_path'] + "*.txt")
    combined = []

    for path in datasets:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(int(config["total_samples"])):
                sample = f.readline()
                data.append(sample)
        data = pick(data,
                    samples_count if not split_len else samples_count // len(datasets))

        for sample in data:
            sample = clear(sample.lower())
            sample = " ".join([i if not i.isdigit() else num2words(i, lang='ru') for i in sample.split()])
            combined.append(sample)
    return combined


def build_dataset():
    data = get_data(True)
    validation_data = get_data(True, int(config['valid_samples_count']))
    dataset = {"train": data, "valid": validation_data}
    with open(config['train_data_path'], 'w', encoding='utf-8') as f:
        dump(dataset, f, ensure_ascii=False)
    try:
        pass
    except:
        return Response(0, "Something went wrong while building dataset")
    return Response(1, "Dataset was built successfully")


def load_dataset():
    with open(config['train_data_path'], 'r', encoding='utf-8') as f:
        return load(f)