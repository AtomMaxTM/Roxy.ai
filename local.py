# from threading import Thread
# from core.voice.VoiceProcessor.stt import STT
# from core.nlu.CommandClassifier.classifier import Classifier
#
# def process(q, classifier):
#     while True:
#         data = q.get()
#         x = classifier(data)
#         print(x)
#
# def main():
#     stt = STT()
#     queue, status = stt.run(debug=True)
#     print(status)
#     classifier = Classifier()
#     print(classifier.load_models())
#
#     thread1 = Thread(target=process, args=(queue, classifier))
#     thread1.start()
#     return stt
from modules.TelegramBot.bot import start
start()