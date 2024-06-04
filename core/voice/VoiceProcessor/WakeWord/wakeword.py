from core.voice.VoiceProcessor.WakeWord.eff_word_net.engine import HotwordDetector
from core.voice.VoiceProcessor.WakeWord.eff_word_net.audio_processing import Resnet50_Arc_loss
from core.scripts.config_manager import get_config

def get_model():
    config = get_config()
    return HotwordDetector(
        hotword=config["wakeword"]["wakeword"],
        model=Resnet50_Arc_loss(),
        reference_file=config["wakeword"]["model_path"],
        threshold=float(config["wakeword"]["threshold"]),
        relaxation_time=2
    )