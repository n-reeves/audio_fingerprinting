import librosa
import numpy as np
import os
import random
from matplotlib import pyplot as plt

##########
####contains general functions used to aid various stages of the project
###########

#converts input single channel wav file to spectogram
def to_spect(x, sr, hop_per=.5, win_sec=.064 ):
    """
    Input
        x np array: numpy array representing wav file
        sr int: sample rate
        hop_per float: perceentage of overlap between consecutive windows
        win_sec float) window length in seconds
    Returns:
        numpy array
    """
    win_len = int(2 ** np.ceil(np.log2(sr * win_sec)))
    hop_len = int(round(win_len*hop_per))

    stft = librosa.stft(x, n_fft=win_len, hop_length=hop_len, win_length=win_len, center=True)
    spect = np.abs(stft)
    return spect 

#loads a file from a given path at a given sample rate
def load_file(file_path,sr):
    """ Takes in file path 
    ln: input file path, optional annotations for training
    out: x (np. array representing wav)
        sr scalar: sample rate
    """
    x, sr = librosa.load(file_path, sr=sr)
    return x, sr


def split_data(db_file_path, q_file_path, test_percent, ran_seed=0):
   #function that intakes file path, and percent of dataset in test set
   #returns a dictionary: {'train':[...],'test':[...] } with train and test keys
    
    #iterates through the files and loads name into list. shuffles list with fixed random state
    db_files = [ file for file in os.listdir(db_file_path) if file.endswith('.wav') ]
    q_files = [ file for file in os.listdir(q_file_path) if file.endswith('.wav') ]
    
    random.seed(ran_seed)
    random.shuffle(db_files)
    
    #splits database files into into train and test lists
    test_size = int(len(db_files)*test_percent)
    db_test_files = db_files[:test_size]
    db_train_files = db_files[test_size:]
    
    #contstruct query sets using the database splits
    q_train_files = []
    for file_db in db_train_files:
        for file_q in q_files:
            if file_db[:-4] in file_q:
                q_train_files.append(file_q)
                
    q_test_files = []
    for file_db in db_test_files:
        for file_q in q_files:
            if file_db[:-4] in file_q:
                q_test_files.append(file_q)
        
    return {'train': {'db':db_train_files, 'q': q_train_files}, 'test':{'db':db_test_files, 'q': q_test_files} }


#converts the predictions and labels to the output format required by CW
def preds_to_file(preds, label_fnames, results_path):
    text_out = ''
    
    for i in range(len(preds)):
        line = label_fnames[i] + '\t' + preds[i][0] + '\t' + preds[i][1]+ '\t' + preds[i][2] + '\n'
        text_out += line
                    
    with open(results_path, 'a') as f:
        f.write(text_out)


#Note: this function was taken from a ECS7013 lab. Included as used to produce 
#report spectrograms
def spec_plot(audio, sr=22050, n_fft=1028, hop_length=256, save_png=False, png_name='test.png'):
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X)) 
    
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    
    if save_png:
        plt.savefig(png_name)
        
    plt.show()

#takes input 2d array of peaks and plots the constellation map
def plot_const(peaks):
    plt.imshow(peaks, cmap='binary', interpolation='nearest', aspect='auto')
    plt.show()
    
    