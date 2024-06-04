import json
import queue
import time
from glob import glob
from random import choice
from threading import Thread

import sounddevice as sd
import vosk
from soundfile import read

from core.scripts.config_manager import get_config
from core.scripts.tools import Response, say
from core.voice.VoiceProcessor.WakeWord.eff_word_net.streams import SimpleMicStream
from core.voice.VoiceProcessor.WakeWord.wakeword import get_model

class VoskModel:
    def __init__(self):
        self.config = get_config()
        self.samplerate = 16000
        self.model = None

    def load_vosk(self):
        if self.model is None:
            vosk_model = vosk.Model(self.config["stt"]["model_path"])
            self.model = vosk.KaldiRecognizer(vosk_model, self.samplerate)
            try:
                pass
            except Exception as e:
                return Response(0, 'Something went wrong while loading vosk stt model', e)
        else:
            return Response(-1, 'Model is already loaded')
        return Response(1, 'Model was loaded successfully')

    def reset(self):
        self.model = None
        return self.load_vosk()


stt_model = VoskModel()
stt_model.load_vosk()


class STT:
    """
    STT(Speech To Text) class realization
    Uses eff_word_net for wakeword detection and vosk stt from speech recognition
    """

    def __init__(self):
        self.config = get_config()
        self.answers = [read(i.replace("\\", "/"))[0] for i in
                        glob("./core/voice/VoiceProcessor/PregeneratedAnswers/roxy_command/*.wav")]
        self.stream = SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.75,
        )
        self.q = queue.Queue()
        self.wakeword_model = None
        self.timeout = int(self.config['stt']['timeout'])
        self.thread = None
        self.__running = False

    def load_wakeword(self) -> Response:
        """
        Method that loads the wakeword model
        :return: Response object
        """
        if self.wakeword_model is None:
            try:
                self.wakeword_model = get_model()
            except Exception as e:
                return Response(0, 'Something went wrong while loading wakeword model', e)
        else:
            return Response(-1, 'Model is already loaded')
        return Response(1, 'Model was loaded successfully')

    def listen(self, query, debug=False):
        def q_callback(indata, frames, time, debug):
            self.q.put(bytes(indata))
        self.stream.start_stream()
        print("Running..." if debug else "", end='\n' if debug else '')
        while True:
            frame = self.stream.getFrame()
            result = self.wakeword_model.scoreFrame(frame)
            if result is None:
                continue
            if result['confidence'] >= float(self.config["wakeword"]["threshold"]):
                print(f"Confidence: {result['confidence']}" if debug else "", end="\n" if debug else "")
                lts = time.time()
                self.stream.close_stream()
                say(choice(self.answers))
                with sd.RawInputStream(samplerate=16000, blocksize=8000, device=1,
                                       dtype='int16', channels=1, callback=q_callback):
                    while time.time() - lts <= self.timeout:
                        data = self.q.get()
                        if stt_model.model.AcceptWaveform(data):
                            res = json.loads(stt_model.model.Result())['text']
                            if res != "":
                                if res == 'стоп':
                                    break
                                print(f"Text: {res}" if debug else "", end="\n" if debug else "")
                                query.put(res)
                                lts = time.time()
                                # else:
                                #     print(f"An error occurred: {status.error}" if debug else "",
                                #           end="\n" if debug else "")

                    self.q.empty()

                self.stream.start_stream()
                print("Listening for wakeword..." if debug else "", end="\n" if debug else "")

    def run(self, callback=None, separate_process: bool = True, debug: bool = False) -> tuple[queue.Queue, Response] | tuple[None, Response]:
        """
        Method that starts speech recognition in parallel or non-parallel mode
        :param callback: Function that will be called after recognition. Must take one argument: text
        :param separate_process: Determines whether speech recognition will be started as a separate process or not
        :param debug: Shows debug information while running
        :return: Response object and queue object if process started successfully
        """
        if self.__running:
            return None, Response(-1, 'STT is already running')
        if callback is not None and not callable(callback):
            return None, Response(0, 'Callback function must be callable')

        wakeword_load_response = self.load_wakeword()
        if wakeword_load_response.status != 1:
            return None, wakeword_load_response
        vosk_load_response = stt_model.reset()
        if vosk_load_response.status != 1:
            return None, vosk_load_response

        if separate_process:
            try:
                q = queue.Queue()
                self.thread = Thread(target=self.listen, args=(q, debug))
                self.thread.start()
            except Exception as e:
                return None, Response(0, 'An error occurred while starting the process', e)
            return q, Response(1, 'STT successfully started in parallel mode')

        try:
            self.__running = True
            self.listen(callback, debug)
        except Exception as e:
            self.__running = False
            return None, Response(0, 'An error occurred while starting STT', e)
        self.__running = False

    def stop(self) -> Response:
        """
        Method that stops the process
        :return: Response object
        """
        if not self.__running:
            return Response(-1, 'STT is not running')
        try:
            self.thread.stop()
        except Exception as e:
            return Response(0, 'An error occurred while starting the process', e)
        self.__running = False
        return Response(1, 'STT successfully started in parallel mode')
