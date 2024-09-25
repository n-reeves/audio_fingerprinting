
import pickle
from .database import DataBase, QuerySet
from .eval import run_fingerprint
from .utilities import preds_to_file
import time

##########
####Top level functions used to create a database of fingerprints and run audio identification
###########

# best performing parameters
best_params = {'f_win_p': 17, 't_win_p': 5, 'f_hop_len_p': 10, 't_hop_len_p': 5, 't_shift_h': 21, 'f_win_h': 7, 't_win_h': 18, 'num_bins': 48}

#used to create a database of fingerprints
def fingerprintBuilder(db_file_path, db_save_loc, sr=22050, params=best_params):
    """
    Args:
        db_file_path (str, optional): file path of folder containg database audio files
        db_save_loc (str, optional): file path to the location the database will be saved in
            #Note: the path should include the file name. The pkl extension is used to save the database object so no extension is needed
        sr (int, optional): _description_. Defaults to 22050.
        params (_type_, optional): _description_. Defaults to best_params.
    """
    database = DataBase(sr=sr, file_path=db_file_path)
    
    print('loading database')
    database.load_data()
    
    print('creating hashes')
    database.create_hash_db(f_win_p=params['f_win_p']
                        ,t_win_p=params['t_win_p']
                        ,f_hop_len_p=params['f_hop_len_p']
                        ,t_hop_len_p=params['t_hop_len_p']
                        ,t_shift_h=params['t_shift_h']
                        ,f_win_h=params['f_win_h'] 
                        ,t_win_h=params['t_win_h'])
    
    #need to make sure db saves correctly
    with open(db_save_loc+'.pkl', 'wb') as f:
        pickle.dump(database, f)
        
    print('database saved: {}'.format(db_save_loc + '.pkl'))

#runs audio identification loop and saves predictions
def audioIdentification(q_path, db_path, results_path, params=best_params, sr=22050):
    """
    Args:
        q_path (str, optional):  path to the folder contain the query wavs.
        db_path (str, optional): path to the saved database. Does not need to include the pkl extension
                #ex: '/path/to/saved/database/file_name' or '/path/to/saved/database/file_name.pkl' work
        results_path (str, optional): location to save the results file. doe
        params (_type_, optional): _description_. Defaults to best_params.
        sr (int, optional): _description_. Defaults to 22050.
    """
    if db_path[-4:] != '.pkl':
        db_path += '.pkl'
        
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    
    start_t = time.time()
    
    queryset = QuerySet(sr=sr, file_path=q_path)
    
    print('loading queryset')
    queryset.load_data()
    
    print('creating query hashes')
    queryset.create_hash_db(f_win_p=params['f_win_p']
                        ,t_win_p=params['t_win_p']
                        ,f_hop_len_p=params['f_hop_len_p']
                        ,t_hop_len_p=params['t_hop_len_p']
                        ,t_shift_h=params['t_shift_h']
                        ,f_win_h=params['f_win_h'] 
                        ,t_win_h=params['t_win_h'])
    
    preds, _, label_fnames = run_fingerprint(database, queryset, num_bins = params['num_bins'])
    
    end_t = time.time()
    ret_t = end_t - start_t
    
    #add missing extension
    if not results_path.endswith('.txt'):
        results_path += '.txt'
    
    preds_to_file(preds, label_fnames,results_path)
    
    print('audio identification complete. Results saved to: {}'.format(results_path))
    print('Creating query hashes and retrieval took: {} seconds'.format(ret_t))
    
