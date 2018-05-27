import CreateAudio
import test_noise as tn
import librosa
import numpy as np
import time
def load_data():
    arr = []
    for i in range(1,11):
        x, sr = librosa.load("../wav/trai_%s.wav" % i)
        mfcc = librosa.feature.mfcc(x, sr)
        arr.append(mfcc)

    for i in range(1,11):
        x, sr = librosa.load("../wav/phai_%s.wav" % i)
        mfcc = librosa.feature.mfcc(x, sr)
        arr.append(mfcc)

    for i in range(1,11):
        x, sr = librosa.load("../wav/tien_%s.wav" % i)
        mfcc = librosa.feature.mfcc(x, sr)
        arr.append(mfcc)

    for i in range(1,11):
        x, sr = librosa.load("../wav/lui_%s.wav" % i)
        mfcc = librosa.feature.mfcc(x, sr)
        arr.append(mfcc)

    for i in range(1,11):
        x, sr = librosa.load("../wav/dung_%s.wav" % i)
        mfcc = librosa.feature.mfcc(x, sr)
        arr.append(mfcc)

    return arr

def sosanh(arr):
    test = CreateAudio.Audio()
    test.createAudio("../wav/test.wav")
    # tn.outfile("../wav/test.wav","../wav/test_1.wav")
    x, sr = librosa.load("../wav/test.wav")
    mfcc = librosa.feature.mfcc(x, sr)
    delta = []
    for i in range(len(arr)):
        delta.append(np.linalg.norm(mfcc - arr[i]))
    min1 = min(delta)
    # print(delta)
    print(min1)
    return int(delta.index(min1)/10)


# arr = load_data()
# while True :
#
#   print(sosanh(arr))
#   time.sleep(5)
