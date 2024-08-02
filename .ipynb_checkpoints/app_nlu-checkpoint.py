from termcolor import colored  # вывод цветных логов (для выделения распознанной речи)
import speech_recognition  # распознавание пользовательской речи (Speech-To-Text)
import wave  # создание и чтение аудиофайлов формата wav
import os  # работа с файловой системой
import torch
from sentence_transformers import SentenceTransformer # библиотека загрузки моделей из ресурса
import requests

import pyaudio
import joblib
import sys

#загрузка в гпу или цпу в зависимости от наличия их на устройстве
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели получения эмбедингов
model = SentenceTransformer("cointegrated/LaBSE-en-ru").to(device) # load bert

# Загрузка класификатора
clf = joblib.load('classifier_labse_proba2.joblib')

#Подключение микрофона
recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone()


"""
Проигрывание речи ответов голосового ассистента
:param text_to_speech: текст, который нужно преобразовать в речь
"""
def play_voice():

    audio_paths = 'audio.wav'
    chunk = 1024
    f = wave.open(audio_paths, "rb")
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)
    # play stream
    frames = []
    while data:
        stream.write(data)
        data = f.readframes(chunk)
        frames.append(data)
    # stop stream
    stream.stop_stream()
    stream.close()
    # close PyAudio
    p.terminate()

# Распознавание аудио через vk speech kit, нужно зарегаться на vk speech kit и оттуда получить токен
def recognize_audio():
    file = "SPEAKER_02.wav"
    with open(file, "rb") as f:
        data = f.read()
    headers = {'Content-Type': 'audio/wave',
               'Authorization': 'Bearer qgNY6LAXE43vA6DCHdT2EaMqGfkEYqyY3oPvf3K15xiu19Z7g'}
    response = requests.post('https://voice.mcs.mail.ru/asr', headers=headers, data=data) #'https://voice.mcs.mail.ru/asr'
    text = response.json()['result']['texts'][0]['text']
    # if os.path.exists(file):
    #     os.remove(file)
    return text

# Запись аудио
def record_audio():
    with microphone:
        recognized_data = ""

        # запоминание шумов окружения для последующей очистки звука от них
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 10, 10)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            return
    
# Синтез аудио через vk speech kit, нужно зарегаться на vk speech kit и оттуда получить токен
def get_speech_from_vk(text):
    headers = {'Content-Type': 'application/json',
               'Authorization': 'Bearer 8aa2d8b48aa2d8b48aa2d8b49789b8eba288aa28aa2d8b4ec01b0aaca885f1807694920'}
    response = requests.get(F'https://voice.mcs.mail.ru/tts?text={text}?', headers=headers)
    with open('audio.raw', 'wb') as wav:
        wav.write(response.content)
    os.system('C:\sox-14-4-2\sox.exe -r 24000 -b 16 -e signed-integer -B -c 1 audio.raw audio.wav')

#Получение ответа пользователю из класификатора
def find_intent(voice_input):
    return clf.predict(model.encode(voice_input).reshape(1, -1))[0]


def recognize_audio_file(file):
    with open(file, "rb") as f:
        data = f.read()
    headers = {'Content-Type': 'audio/wave',
               'Authorization': 'Bearer qgNY6LAXE43vA6DCHdT2EaMqGfkEYqyY3oPvf3K15xiu19Z7g'}
    response = requests.post('https://voice.mcs.mail.ru/asr', headers=headers, data=data) #'https://voice.mcs.mail.ru/asr'
    text = response.json()['result']['texts'][0]['text']
    return text

# Основной цикл 
'''
while True:
    record_audio()
    voice_input = recognize_audio()
    print(voice_input)
    intent = find_intent(voice_input)
    print(colored(intent, "blue"))
    get_speech_from_vk(intent)
    play_voice()
'''

voice_input = recognize_audio_file('ch1_1.wav')
print('chanel_1: \n', voice_input)
print()
voice_input = recognize_audio_file('ch2_1.wav')
print('chanel_2: \n', voice_input)
