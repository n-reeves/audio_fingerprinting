import random
import numpy as np
from .database import DataBase, QuerySet

##########
####Contains functions used to evaluate the fingerprinting algorithm as well the primary fingerprinting loop
###########

#function that produces evaluation summary dictionary with mean precision, recall, and f1 scores
def eval_summary(preds, labels):
    #input: preds: list of lists of predictions
            #labels: list of labels
    #output: eval_dict: dictionary with rank as key and mean rank precision, recall, and f1 as key value pairs
    
    #calc the percent of querys that were correctly matched in top n results
    eval_dict = {}
    relevant = 1 #assuming that the search is based off of a single relevant document
    
    eval_range = len(preds[0]) 
    
    #iterate through each ranking (1-3) and calculate mean precision, recall, and f1
    for rank in range(1,eval_range+1):
        prec_vals = []
        rec_vals = []
        f_vals = []
        for i, pred in enumerate(preds):
            if labels[i] in pred[:rank]:
                found = 1
            else:
                found = 0
            prec = found/rank #number of relevant documents returned, over total number of documents returned
            rec = found/relevant #number of relevant documents returned, over total number of relevant documents
            if prec+rec == 0:
                f = 0
            else:
                f = 2*(prec*rec)/(prec+rec)

            prec_vals.append(prec)
            rec_vals.append(rec)
            f_vals.append(f)
        
        eval_dict[rank] = {'precision':np.mean(prec_vals)
                           ,'recall':np.mean(rec_vals)
                           ,'f1':np.mean(f_vals)}
          
        print('Rank:{0}\nPrecision:{1}\nRecall:{2}\nF1:{3}'
              .format(rank, np.mean(prec_vals), np.mean(rec_vals), np.mean(f_vals)))
    return eval_dict


#function that takes the bin of hash pairs and returns a score based on the bin of the
#histogram that has the maximum number of offsets
def bin_to_hist(bin, num_bins=100):
    #input: array of tuples. Each element has two values
    #0 : the time offset of the hash in the database
    #1: the time offset of the hash in the query
    bin_np = np.array(bin)
    
    db_t = bin_np[:,0]
    q_t = bin_np[:,1]
    
    #assume that there should be a linear replationship between the time of sets of query and db hashes
    lam = db_t - q_t
    lam_hist = np.histogram(lam, num_bins)
    
    bin_counts = lam_hist[0]
    score = np.max(bin_counts) #how to account for records that are 
    
    return score


#takes a database and query set and returns a list of predictions and a list of labels
def run_fingerprint(database, query_set, num_bins=100):
    print('Starting identification loop')
    labels = []
    label_fnames = []
    preds = []
    hash_db = database.get_hash_db()

    for q_key in query_set.get_keys():
        top_three = [(0, ''), (0, ''), (0, '')]
        label = q_key.split('-')[0]
        labels.append(label)
        label_fnames.append(q_key)
        
        query_hashes = query_set.get_hash_db()[q_key] #dictionary of hash:time key val pairs
        bins = {} # key: query hash, value: list lists with elements = [db time dif , query time dif]
        # iterate through each hash produced from the query
        for q_hash in query_hashes:
            #if the query hash is in the hashes in the database, add the paired hashes anchor time to the bin
            if q_hash in hash_db.keys():
                hash_matches = hash_db[q_hash] #dictionary of wav name:offset pairs
                for db_key in hash_matches.keys():
                    db_time = hash_matches[db_key]
                    query_time = query_hashes[q_hash]
                    time_pair = [db_time,query_time]
                    
                    if db_key in bins.keys():
                        bins[db_key].append(time_pair)
                    else:
                        bins[db_key] = [time_pair]
       
        #use offset time pairs to draw association between db entry and query
        #if query matches, expect positive linear relationship between times
        #of hash anchor points in query and times of hash anchor point in db record
        for db_key in bins.keys():
            score = bin_to_hist(bins[db_key],num_bins=num_bins)
            if score > top_three[2][0]:
                top_three[2] = (score, db_key[:-4])
                top_three.sort(reverse=True)
        top_three = [x[1] for x in top_three]
        
        preds.append(top_three)
    
    return preds, labels, label_fnames


#function used to evaluate a set of parameters
#creates database and query sets and then runs fingerprinting algorithm
def hp_test(in_params, keys, db_path='./Data/database_recordings/', query_path='./Data/query_recordings/'):
    database = DataBase(sr=22050, file_path=db_path, keys=keys['db'])

    query_set = QuerySet(sr=22050, file_path=query_path, keys=keys['q'])
    print('loading data')
    database.load_data()
    query_set.load_data()
    
    print('creating hashes')
    database.create_hash_db(f_win_p=in_params['f_win_p']
                        ,t_win_p=in_params['t_win_p']
                        ,f_hop_len_p=in_params['f_hop_len_p']
                        ,t_hop_len_p=in_params['t_hop_len_p']
                        ,t_shift_h=in_params['t_shift_h']
                        ,f_win_h=in_params['f_win_h'] 
                        ,t_win_h=in_params['t_win_h'])
    
    query_set.create_hash_db(f_win_p=in_params['f_win_p']
                        ,t_win_p=in_params['t_win_p']
                        ,f_hop_len_p=in_params['f_hop_len_p']
                        ,t_hop_len_p=in_params['t_hop_len_p']
                        ,t_shift_h=in_params['t_shift_h']
                        ,f_win_h=in_params['f_win_h'] 
                        ,t_win_h=in_params['t_win_h'])
    
    preds, labels, _ = run_fingerprint(database, query_set, num_bins=in_params['num_bins'])
    
    eval_dict = eval_summary(preds,labels)
    return eval_dict

#loop used to evaluate randomly selected parameters
def random_grid_search(keys, epochs=30, db_path='./Data/database_recordings/', q_path='./Data/query_recordings/'):
    #keys expects format {'db':list of db keys, 'q':list of query keys}
    num_bins_vals = np.arange(5, 50)
    f_win_vals = np.arange(5, 20)
    t_win_vals = np.arange(5, 20)
    t_shift_vals = np.arange(1, 100)
    
    eval_dicts = []

    #randomly sample from the hyperparameter space
    for i in range(epochs):
        #ranomly sample each hyperparameter from the range and pass them in to hp test loop
        f_win_p = random.choice(f_win_vals)
        t_win_p = random.choice(t_win_vals)
        num_bins = random.choice(num_bins_vals)
        
        f_win_h = random.choice(f_win_vals)
        t_win_h = random.choice(t_win_vals)
        t_shift_h = random.choice(t_shift_vals)
        
        
        f_hop_len_vals = np.arange(5, f_win_p+1)
        t_hop_len_vals = np.arange(5, t_win_p+1)
        f_hop_len_p = random.choice(f_hop_len_vals)
        t_hop_len_p = random.choice(t_hop_len_vals)
        
        in_params = {'f_win_p':f_win_p
                ,'t_win_p':t_win_p
                ,'f_hop_len_p':f_hop_len_p
                ,'t_hop_len_p':t_hop_len_p
                ,'t_shift_h':t_shift_h
                ,'f_win_h':f_win_h
                ,'t_win_h':t_win_h
                ,'num_bins':num_bins}

        print(in_params)
        
        eval_dict = hp_test(in_params, keys, db_path, q_path)
        eval_dicts.append(eval_dict)
        
    return eval_dicts
    