import torch
from core.scripts.tools import get_config

vad_config = get_config()['vad']

def init_jit_model(model_path: str, device):
    model = torch.jit.load(model_path, map_location=torch.device(device))
    model.eval()
    return model


model = init_jit_model(vad_config['vad_model_path'], device=vad_config['device'])


class VAD:
    def __init__(self, model, sr=16000, threshold=0.6):
        self.model = model
        self.sr = sr
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, x):
        response = self.model(torch.tensor(x, dtype=torch.float32), 16000).item()
        return response
