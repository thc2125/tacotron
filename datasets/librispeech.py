import shutil

import os.path
from pathlib import Path
from pydub import AudioSegment

import numpy as np

from util import audio

def normalize(data_dir, normalized_dir):
    wav2flac(data_dir, normalized_dir)
    dup_txt(data_dir, normalized_dir)

def wav2flac(data_dir, normalized_dir):
    flac_files = sorted(Path(data_dir).glob('**/*.flac'))

    for flac_file in flac_files:
        wav_dir = Path(normalized_dir) / flac_file.relative_to(data_dir).parent
        if not wav_dir.exists():
            wav_dir.mkdir(parents=True)

        AudioSegment.from_file(str(flac_file), 'flac').export(
                str(wav_dir /
                    (flac_file.stem + '.wav')),
                format='wav')

def dup_txt(data_dir, normalized_dir):
    txt_files = sorted(Path(data_dir).glob('**/*.txt'))
    for txt_file in txt_files:
        new_dir = Path(normalized_dir) / txt_file.relative_to(data_dir).parent
        if not new_dir.exists():
            new_dir.mkdir(parents=True)

def preprocess(data_dir, normalized_dir):
    # Ingest the text file:
    #normalize(data_dir,  normalized_dir)
    wav_files = sorted(Path(data_dir).glob('**/*.wav'))
    train_tuples = []
    for wav_file in wav_files:
        # Get the transcript
        text_filepath = (wav_file.parent
                     / ('-'.join(wav_file.stem.split('-')[:-1])
                       + '.trans.txt'))
        text = ''
        with text_filepath.open() as text_file:
            for line in text_file:
                split_line = line.split()
                if split_line[0] == wav_file.name:
                    text = ' '.join(split_line[1:])
                    break
        
        wav = audio.load_wav(str(wav_file))

        spectrogram = audio.spectrogram(wav).astype(np.float32)
        spectrogram_filename = (wav_file.stem + '_spectrogram')
        mel_spectrogram_filename = (wav_file.stem + '_mel_spectrogram')

        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
        np.save(os.path.join(normalized_dir,spectrogram_filename),
                spectrogram.T,
                allow_pickle=False)
        np.save(os.path.join(normalized_dir,mel_spectrogram_filename),
                mel_spectrogram.T,
                allow_pickle=False)
        train_tuples.append((spectrogram_filename, mel_spectrogram_filename,
                spectrogram.shape[1],
                text))
        print('DOne??')
        with open(os.path.join(normalized_dir, 'train.txt'), 'w') as train_file:
            for train_tuple in train_tuples:
                train_file.write('|'.join([str(x) for x in train_tuple]) + '\n')
