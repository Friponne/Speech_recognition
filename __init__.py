
import vosk, json
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import joblib
import torch

model = vosk.Model("vosk-model-ru-0.10")

#загрузка в гпу или цпу в зависимости от наличия их на устройстве
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели получения эмбедингов
model_emb = SentenceTransformer("cointegrated/LaBSE-en-ru").to(device) # load bert

# Загрузка класификатора
clf = joblib.load('classifier_labse_proba2.joblib')

FRAME_RATE = 16000
CHANNELS = 1

rec = vosk.KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)


def wav_to_txt(file_name_wav):
    audio = AudioSegment.from_file(file=file_name_wav,format="wav")
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_frame_rate(FRAME_RATE)

    rec.AcceptWaveform(audio.raw_data)
    result = rec.Result()
    text = json.loads(result)
    print(type(text))
    print('________________________________')
    print(text)
    text = text["text"]

    file_name_txt = 'data.txt'

    # Записываем результат в файл "data.txt"
    with open(file_name_txt, 'w') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)

#Получение ответа пользователю из класификатора
def find_intent(voice_input):
    return clf.predict(model_emb.encode(voice_input).reshape(1, -1))[0]

wav_to_txt('test_1.wav')
print(find_intent('data.txt'))
