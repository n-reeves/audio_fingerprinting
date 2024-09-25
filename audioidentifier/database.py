import os
from .utilities import load_file, to_spect
from .hash import peak_picking, get_pair_hash

##########
####Contains classes used to store database and query hashes and handel data processing
###########


# create database class that inits with a set of keys, a sample rate, and a file path
# loads the files into an instance variable array
class DataSet:
    def __init__(self, sr, file_path,keys):
        """ Takes in keys, sample rate, and file path
        ln: keys, sample rate, file path
        out: None
        """
        self.keys = keys
        self.sr = sr
        self.file_path = file_path
        self.data = {}
        
        if len(self.keys) == 0:
            self.keys = os.listdir(file_path)
            self.keys = [key for key in self.keys if key.endswith('.wav')]
        
    def load_data(self):
        """ Takes in None
        ln: None
        out: None
        """
        for key in self.keys:
            path = os.path.join(self.file_path, key)
            x, _ = load_file(path, self.sr)
            self.data[key] = {'wav':x}           
            
    def get_data(self):
        return self.data
    
    def get_keys(self):
        return self.keys
    
    def get_sr(self):
        return self.sr

#inherits from dataset, class used to store database hashes
class DataBase(DataSet):
    def __init__(self, sr, file_path, keys=[]):
        super().__init__(sr, file_path, keys)
        self.hash_db = {}
        #hash db: key: hash tuple, contains dict of wav_nam:offset pairs
    
    def create_hash_db(self
                        ,f_win_p=5
                        ,t_win_p=5
                        ,f_hop_len_p=20
                        ,t_hop_len_p=20
                        ,t_shift_h=50
                        ,f_win_h=50
                        ,t_win_h=50):
        for key in self.keys:
            wav = self.data[key]['wav']
            
            #convert wav to spectogram
            spect = to_spect(wav, self.sr, hop_per=.5, win_sec=.064 )
            self.data[key]['wav'] = None # clear memory
            
            #get peaks from spectogram
            peaks = peak_picking(spect
                                ,f_win=f_win_p
                                ,t_win=t_win_p
                                ,f_hop_len=f_hop_len_p
                                ,t_hop_len=t_hop_len_p)
            
            #get hash offset pairs from peaks
            hashes = get_pair_hash(peaks
                                   ,t_shift=t_shift_h
                                   ,f_win=f_win_h
                                   ,t_win=t_win_h)
            
            #iterate through file hashes
            for entry in hashes:
                hash = entry[0]
                offset = entry[1]
                #if hash not in hash_db, add a dictionary with key:filename, value:offset
                if not hash in self.hash_db:
                    self.hash_db[hash] = {key:offset}
                else: #otherwsie add hash to existing dictionary
                    self.hash_db[hash][key] = offset
                    
    def get_hash_db(self):
        return self.hash_db


class QuerySet(DataSet):
    def __init__(self, sr, file_path,keys=[]):
        super().__init__(sr, file_path, keys)
        #query_hash_db: key: wav filename, value:  dict with hash offset key value pairs
        self.query_hash_db = {}
    
    def create_hash_db(self
                        ,f_win_p=5
                        ,t_win_p=5
                        ,f_hop_len_p=20
                        ,t_hop_len_p=20
                        ,t_shift_h=50
                        ,f_win_h=50
                        ,t_win_h=50):
        
        #similar to database hash creation, but filenames are keys and hash offset pairs are key value pairs
        for key in self.keys:
            wav = self.data[key]['wav']
            spect = to_spect(wav, self.sr, hop_per=.5, win_sec=.064 )
            peaks = peak_picking(spect
                                ,f_win=f_win_p
                                ,t_win=t_win_p
                                ,f_hop_len=f_hop_len_p
                                ,t_hop_len=t_hop_len_p)
            
            hashes = get_pair_hash(peaks
                                   ,t_shift=t_shift_h
                                   ,f_win=f_win_h
                                   ,t_win=t_win_h)
            
            self.query_hash_db[key] = {}
            for entry in hashes:
                hash = entry[0]
                offset = entry[1]
                self.query_hash_db[key][hash] = offset
    
    def get_hash_db(self):
        return self.query_hash_db