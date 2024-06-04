from core.voice.VoiceProcessor.WakeWord.eff_word_net.generate_reference import generate_reference_file
from core.scripts.tools import say
import sounddevice as sd
import soundfile as sf
import os
import librosa

pro, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/say_wakeword.wav")
rep, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/repeat_wakeword.wav")
gen, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/generating_model.wav")
suc, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/model_successfully_generated.wav")
err, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/something_went_wrong.wav")
exi, _ = sf.read("./Client/Voice/PregeneratedAnswers/wakeword/model_exists.wav")
sr = 44100



def listen_for_voice(path: str, num: int, debug: bool= False, kostyl:bool = True, clones:int = 10):
    if os.path.isfile(f"{path}/user{num}0.wav"):
        if debug:
            print(f"user{num}.wav already exists")
        return 1
    say(pro if kostyl else rep)
    data = sd.rec(int(sr * 2.5), samplerate=sr,
                  channels=1, blocking=True)
    clean_data, _ = librosa.effects.trim(data, top_db=50)
    for i in range(clones+1):
        sf.write(f"{path}/user{num}{i}.wav", clean_data, sr)
    return 0

def prepare_audio(wakeword, wakeword_dir, user_samples, clones:int = 20):
    if not os.path.exists(wakeword_dir):
        os.mkdir(wakeword_dir)
    val = True
    for i in range(user_samples):
        val = listen_for_voice(wakeword_dir, i, kostyl=bool(val), clones=clones)
    say(gen)

def save_user_voice(wakeword: str, model_name: str, user_samples: int=2, clones: int=20):
    if os.path.isfile(f"./Client/Voice/models/wakeword_models/{model_name}_ref.json"):
        say(exi)
        return 0
    wakeword_dir = f"./Client/Voice/WakeWord/TrainAudioFiles/{model_name}"
    prepare_audio(wakeword, wakeword_dir, user_samples)
    try:
        generate_reference_file(wakeword_dir, "./Client/Voice/models/wakeword_models/", model_name)
    except Exception as e:
        print(f"Error: {e}")
        say(err)
    else:
        say(suc)