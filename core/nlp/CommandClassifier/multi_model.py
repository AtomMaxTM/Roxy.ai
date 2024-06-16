from glob import glob
from typing import Any
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from core.nlp.CommandClassifier.classifier_tools import load_config, config, tokenize, OneHot

input_size = int(config['input_size'])
hidden_size = int(config['hidden_size'])
num_heads = int(config['num_heads'])
batch_size = int(config['batch_size'])

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
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = MultiHeadAttention(num_heads, hidden_size // num_heads, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.attention(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.softmax(x)
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
    configs = [i.replace("\\", "/") for i in glob(config['commands_path'], recursive=True)]
    labels = []
    samples = []
    labels_save = {}
    labels_len = len(configs)
    for i in configs:
        temp = load_config(i)
        labels_save[temp[0]] = temp[2]
        for sample in temp[1]:
            labels.append(temp[0])
            samples.append(sample)

    return samples, labels, labels_len, labels_save


def validate(model):
    val_data = (
    ('можешь закрыть окно', 1), ('выключи текущую вкладку', 1), ('закрой это окно', 1), ('закрой текущую вкладку', 1),
    ('можешь выключить браузер', 1), ('открой новую страницу', 0), ('включи новую вкладку', 0),
    ('открой новое окно', 0), ('можешь открыть новую страницу', 0), ('открой вкладку сверху', 0),
    ('включи последнюю вкладку', 0), ('можешь открыть новое окно', 0), ('закрой текущую страницу', 1),
    ('закрой последнюю вкладку', 1), ('выключи текущую вкладку браузера', 1), ('можешь закрыть текущее окно', 1),
    ('включи новую страницу браузера', 0), ('открой новую вкладку сверху', 0), ('можешь закрыть это окно', 1),
    ('открой новую вкладку в браузере', 0))
    pred = []
    true = []

    model.eval()
    for seq, label in [i for i in val_data]:
        output = model(tokenize(seq)).squeeze()
        pred.append(torch.argmax(output).item())
        true.append(label)
    print(f"True: {true}\nPred: {pred}")

    return accuracy_score(true, pred)


def create_model() -> tuple[Classifier, dict[str, list[Any] | int | dict[Any, Any] | Any]]:
    samples, labels, labels_len, labels_save = get_data()

    enc = OneHot(labels_len, dtype=torch.float32)

    labels = [enc(i) for i in labels]

    embeddings = torch.stack(tokenize(samples))
    dataset = TensorDataset(embeddings,
                            torch.stack(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_hist = []

    accuracy_true, accuracy_pred = [], []

    model = Classifier(input_size, hidden_size, labels_len, num_heads)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in tqdm(range(int(config["epochs"])), total=int(config["epochs"]), desc="Epoch"):
        for batch in dataloader:
            optimizer.zero_grad()
            for sample, label in zip(batch[0], batch[1]):

                output = model(sample).squeeze()

                accuracy_pred.append(torch.argmax(output).item())
                accuracy_true.append(torch.argmax(label).item())

                loss = criterion(output, label)
                loss.backward()
                loss_hist.append(loss.item())
            optimizer.step()

    train_accuracy = accuracy_score(accuracy_true, accuracy_pred)
    # validation_accuracy = validate(model)

    params = {
        "epochs": int(config["epochs"]),
        "num_classes": labels_len,
        "num_heads": num_heads,
        "final_loss": loss_hist[-1],
        "labels": labels_save,
        "train_accuracy": train_accuracy,
        # "validation_accuracy": validation_accuracy,
        "loss_hist": loss_hist
    }
    model.eval()

    return model, params

