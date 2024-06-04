import json
from dataclasses import dataclass
from typing import Any

import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from core.nlu.CommandClassifier.classifier_tools import tokenize, Filter
from core.nlu.CommandClassifier.multi_model import create_model as multi
from core.nlu.CommandClassifier.binary_model import create_model as binary
from core.scripts.config_manager import get_config
from core.scripts.tools import Response

font = {
    'family': 'serif',
    'color': 'darkgrey',
    'weight': 'normal',
    }

@dataclass
class Binary:
    model: any = None
    epochs: int = None
    loss: float = None
    attention_heads: int = None
    loss_hist: list = None
    is_model_loaded: bool = None
    accuracy: float = None
    config: any = None

    def loss_plot(self, gs):
        if not self.is_model_loaded:
            return Response(0, "Model is not loaded")
        ax = plt.subplot(gs[0, 1])
        ax.plot(self.loss_hist)
        ax.set_title(f"Binary Classifier\n\n\n", color='darkgreen')
        ax.set_xlabel("Epochs", fontdict=font, fontsize=18)
        ax.set_ylabel("Loss", fontdict=font, rotation=0, fontsize=18)
        return ax

    def save(self) -> Response:
        if not self.is_model_loaded:
            return Response(0, "Model is not loaded")
        try:
            torch.save(self.model, self.config['model_path'])
            params = {
                      "epochs": self.epochs,
                      "num_heads": self.attention_heads,
                      "final_loss": self.loss,
                      "train_accuracy": self.accuracy,
                      "loss_hist": self.loss_hist
                      }
            with open(self.config['model_params_path'], 'w', encoding="utf-8") as f:
                json.dump(params, f)
        except Exception as e:
            return Response(0, "Something went wrong while saving the model", e)
        return Response(1, "Model saved successfully")

    def load(self) -> Response:
        if self.is_model_loaded:
            return Response(-1, "Model is already loaded")
        try:
            m = torch.load(self.config['model_path'])
            with open(self.config['model_params_path'], 'r', encoding='utf-8') as f:
                temp = json.load(f)
        except Exception as e:
            return Response(0, "Something went wrong while loading binary the model", e)
        self.model = m
        self.epochs = temp['epochs']
        self.loss = temp['final_loss']
        self.loss_hist = temp["loss_hist"]
        self.attention_heads = temp['num_heads']
        self.accuracy = temp['train_accuracy']
        self.model.eval()
        self.is_model_loaded = True
        return Response(1, "Model loaded successfully")

    def train(self) -> Response:
        try:
            model, params = binary()
        except Exception as e:
            return Response(0, "Something went wrong while training the model", e)
        self.epochs = params['epochs']
        self.loss = params['final_loss']
        self.loss_hist = params["loss_hist"]
        self.attention_heads = params['num_heads']
        self.accuracy = params['train_accuracy']
        self.is_model_loaded = True
        self.model = model
        self.model.eval()
        return Response(1, 'Model trained successfully')

    def reset(self) -> Response:
        self.model = None
        self.epochs = None
        self.loss = None
        self.attention_heads = None
        self.loss_hist = None
        self.is_model_loaded = False
        self.accuracy = None
        self.config = None
        return Response(1, 'Model was reset successfully')

    def __call__(self, text) -> tuple[None, Response] | tuple[bool | Any, Response]:
        if not self.is_model_loaded:
            return None, Response(-1, 'Binary model is not loaded')
        try:
            out = self.model(*(tokenize(text)))
        except Exception as e:
            return None, Response(0, 'Something went wrong while using binary model', e)
        return out.item() > float(self.config['threshold']), Response(1)


@dataclass
class Multi:
    model: any = None
    epochs: int = None
    loss: float = None
    attention_heads: int = None
    loss_hist: list = None
    num_classes: int = None
    is_model_loaded: bool = None
    labels: dict = None
    accuracy: float = None
    config: any = None
    filter = Filter()

    def loss_plot(self, gs) -> Response | Any:
        if not self.is_model_loaded:
            return Response(0, "Model is not loaded")
        ax = plt.subplot(gs[0, 0])
        ax.plot(self.loss_hist)
        ax.set_title(f"Multiclass Classifier\n\n\n", color='purple')
        ax.set_xlabel("Epochs", fontdict=font, fontsize=18)
        ax.set_ylabel("Loss", fontdict=font, rotation=0, fontsize=18)
        return ax

    def save(self) -> Response:
        if not self.is_model_loaded:
            return Response(0, "Model is not loaded")
        try:
            torch.save(self.model, self.config['model_path'])
            params = {
                      "epochs": self.epochs,
                      "num_classes": self.num_classes,
                      "num_heads": self.attention_heads,
                      "final_loss": self.loss,
                      "labels": self.labels,
                      "train_accuracy": self.accuracy,
                      "loss_hist": self.loss_hist
                      }
            with open(self.config['model_params_path'], 'w', encoding="utf-8") as f:
                json.dump(params, f)
        except Exception as e:
            return Response(0, "Something went wrong while saving the model", e)
        return Response(1, "Model saved successfully")

    def load(self) -> Response:
        if self.is_model_loaded:
            return Response(-1, "Model is already loaded")
        try:
            m = torch.load(self.config['model_path'])
            with open(self.config['model_params_path'], 'r', encoding='utf-8') as f:
                temp = json.load(f)
        except Exception as e:
            return Response(0, "Something went wrong while loading the model", e)
        self.model = m
        self.epochs = temp['epochs']
        self.loss = temp['final_loss']
        self.loss_hist = temp["loss_hist"]
        self.num_classes = temp['num_classes']
        self.attention_heads = temp['num_heads']
        self.accuracy = temp['train_accuracy']
        self.labels = temp['labels']
        self.model.eval()
        self.is_model_loaded = True
        return Response(1, "Model loaded successfully")

    def train(self) -> Response:
        try:
            model, params = multi()
        except Exception as e:
            return Response(0, "Something went wrong while training the model", e)
        self.labels = params['labels']
        self.epochs = params['epochs']
        self.loss = params['final_loss']
        self.loss_hist = params["loss_hist"]
        self.num_classes = params['num_classes']
        self.attention_heads = params['num_heads']
        self.accuracy = params['train_accuracy']
        self.is_model_loaded = True
        self.model = model
        self.model.eval()
        return Response(1, 'Model trained successfully')

    def reset(self) -> Response:
        self.model = None
        self.epochs = None
        self.loss = None
        self.attention_heads = None
        self.loss_hist = None
        self.num_classes = None
        self.is_model_loaded = False
        self.labels = None
        self.accuracy = None
        self.config = None
        self.filter.update_filters()
        return Response(1, 'Model was reset successfully')

    def __call__(self, text) -> tuple[None, Response] | Response | tuple[Any, Response]:
        if not self.is_model_loaded:
            return None, Response(-1, 'Multiclass model is not loaded')
        try:
            if int(self.config['filter']):
                self.filter.use_spacy = int(self.config['filter_use_spacy'])
                text = self.filter(text)
            out = self.model(*(tokenize(text)))
            res = self.labels[str(torch.argmax(out).item())]
        except Exception as e:
            return None, Response(0, 'Something went wrong while using multiclass model', e)

        return res, Response(1)


@dataclass
class Answer:
    type: int
    value: str = None


class Classifier:
    def __init__(self):
        config = get_config()

        self.__multi = Multi()
        self.__binary = Binary()

        self.__multi.config = config['multiclass_classifier']
        self.__binary.config = config['binary_classifier']

    def save_models(self) -> Response:
        multi_save_response = self.__multi.save()
        if multi_save_response.status != 1:
            return multi_save_response
        bin_save_response = self.__binary.save()
        if bin_save_response.status != 1:
            return bin_save_response

        return Response(1, "Models were saved successfully")

    def train_models(self) -> Response:
        if self.__multi.is_model_loaded:
            return Response(-1, "Model is already loaded")
        multi_train_response = self.__multi.train()
        if multi_train_response.status != 1:
            return multi_train_response

        binary_train_response = self.__binary.train()
        if binary_train_response.status != 1:
            return binary_train_response

        return Response(1, "Models were trained successfully")

    def load_models(self) -> Response:
        multi_load_response = self.__multi.load()
        if multi_load_response.status != 1:
            return multi_load_response
        bin_load_response = self.__binary.load()
        if bin_load_response.status != 1:
            return bin_load_response

        return Response(1, "Models were loaded successfully")

    def reset(self) -> Response:
        self.__multi.reset()
        self.__binary.reset()
        self.reset_configs()
        return Response(1, "Models were reset successfully")

    def reset_configs(self) -> Response:
        config = get_config()
        self.__multi.config = config['multiclass_classifier']
        self.__binary.config = config['binary_classifier']
        return Response(1, "Models configs were reset successfully")

    def plot(self) -> Response:
        if not self.__multi.is_model_loaded and not self.__binary.is_model_loaded:
            return Response(-1, 'Models aren\'t loaded')
        f = plt.figure(figsize=(7, 7))
        gs = GridSpec(1, 2, figure=f)
        ax = self.__binary.loss_plot(gs)
        ax1 = self.__multi.loss_plot(gs)
        plt.show()


    def __call__(self, text) -> Answer | Response:
        if not self.__multi.is_model_loaded and not self.__binary.is_model_loaded:
            return Response(-1, 'Models aren\'t loaded')
        bin, bin_res = self.__binary(text)

        if bin_res.status in [-1, 0]:
            return bin_res
        if not bin:
            return Answer(0, 'Request to LLM')

        mul, mul_res = self.__multi(text)
        if mul_res.status in [-1, 0]:
            return mul_res
        return Answer(1, mul)
