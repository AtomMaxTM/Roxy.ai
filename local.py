# from core.voice.VoiceProcessor.vad import VAD, model
# import numpy as np
# import sounddevice as sd
# import matplotlib.pyplot as plt
# vad = VAD(model, 16000)
# print('Listening...')
# data = (sd.rec(int(16000 * 5), samplerate=16000, channels=1, blocking=True))
# print('... Captured')
#
# def split_data(wav):
#     result = []
#     for i in range(len(wav)//512):
#         result.append(wav[i*512:(i+1)*512])
#     return np.array(result)
# splited = split_data(data.flatten())
# hist = [vad(i) for i in splited]
# plt.plot(hist)
# plt.show()

