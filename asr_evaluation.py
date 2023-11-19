import os
import logging

import deepspeech
import jiwer
import soundfile as sf
import numpy as np
from unidecode import unidecode
import librosa

def evaluate(testset, audio_directory):
    model = deepspeech.Model('deepspeech-0.7.0-models.pbmm')
    model.enableExternalScorer('deepspeech-0.7.0-models.scorer')
    predictions = []
    targets = []
    wers = []
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    for i, datapoint in enumerate(testset):
        logging.info(i)
        audio, rate = sf.read(os.path.join(audio_directory,f'example_output_{i}.wav'))
        if rate != 16000:
            audio = librosa.resample(y=audio, orig_sr=rate, target_sr=16000)
        assert model.sampleRate() == 16000, 'wrong sample rate'
        audio_int16 = (audio*(2**15)).astype(np.int16)
        text = model.stt(audio_int16)
        predictions.append(text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)

        target = transformation(target_text)
        pred = transformation(text)
        if target == '' or pred == '':
            continue
        logging.info(f'targets: {target}')
        logging.info(f'predictions: {pred}')
        curr_wer = jiwer.wer(target, pred)
        logging.info(f'wer: {curr_wer}')
        wers.append(curr_wer)

    logging.info(f'avg wer: {sum(wers) / len(wers)}')

    # transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    # targets = transformation(targets)
    # predictions = transformation(predictions)
    # logging.info(f'targets: {targets}')
    # logging.info(f'predictions: {predictions}')
    # logging.info(f'wer: {jiwer.wer(targets, predictions)}')
    