from glob import glob
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from core.models.nlp.command.bin_dataset_generator.merge_datasets import load_dataset
from core.nlp.CommandClassifier.classifier_tools import load_config, tokenize, augment_binary
from core.nlp.CommandClassifier.classifier_tools import config as cfg
from core.scripts.config_manager import get_config
    
config = get_config()['binary_classifier']
commands_path = cfg['commands_path']
input_size = int(config['input_size'])
hidden_size = int(config['hidden_size'])
num_heads = int(config['num_heads'])
batch_size = int(config['batch_size'])
augment = bool(int(config['augment']))
augment_chance = int(config['augment_chance'])

device = torch.device(
(
    f'cuda:{torch.cuda.current_device()}'
    if torch.cuda.is_available()
    else 'cpu')
    if config['device'] == "auto"
    else config['device']
)

torch.set_default_device(device)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(Classifier, self).__init__()
        self.lnorm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = MultiHeadAttention(num_heads, hidden_size//num_heads, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lnorm(x)
        _, (x, _) = self.lstm(x)
        x = self.attention(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, head_size, in_size):
        super(AttentionHead, self).__init__()
        self.head_size = head_size
        self.key = nn.Linear(in_size, head_size, bias=False)
        self.query = nn.Linear(in_size, head_size, bias=False)
        self.value = nn.Linear(in_size, head_size, bias=False)

    def forward(self, x):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, in_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, in_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


def get_data():
    commands = [i.replace("\\", "/") for i in glob(commands_path, recursive=True)]
    dataset = load_dataset()
    labels = []
    samples = []

    for i in dataset['train']:
        labels.append(torch.tensor(0, dtype=torch.float32))
        samples.append(i)

    for i in commands:
        temp = load_config(i)
        if len(temp[3]) != 0 and augment:
            augmented = augment_binary(temp[1], temp[3], augment_chance)
            for j in augmented:
                labels.append(torch.tensor(1, dtype=torch.float32))
                samples.append(j)
        else:
            for j in temp[1]:
                labels.append(torch.tensor(1, dtype=torch.float32))
                samples.append(j)

    return samples, labels


def create_model() -> tuple[Classifier, dict[str, list[Any] | int | dict[Any, Any] | Any]]:

    samples, labels = get_data()

    bert_embeddings = torch.stack(tokenize(samples))
    dataset = TensorDataset(bert_embeddings,
                            torch.stack(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_hist = []

    accuracy_true, accuracy_pred = [], []

    model = Classifier(input_size, hidden_size, num_heads)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in tqdm(range(int(config["epochs"])), total=int(config["epochs"]), desc="Epoch"):
        for batch in dataloader:
            optimizer.zero_grad()
            for sample, label in zip(batch[0], batch[1]):

                output = model(sample).squeeze()

                accuracy_pred.append(round(output.item()))
                accuracy_true.append(label.item())

                loss = criterion(output, label)
                loss.backward()
                loss_hist.append(loss.item())
            optimizer.step()
    train_accuracy = accuracy_score(accuracy_true, accuracy_pred)

    params = {
        "epochs": int(config["epochs"]),
        "num_heads": num_heads,
        "final_loss": loss_hist[-1],
        "train_accuracy": train_accuracy,
        "loss_hist": loss_hist
    }
    model.eval()

    return model, params
