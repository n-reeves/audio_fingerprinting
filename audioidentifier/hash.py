import numpy as np

##########
####Contains functions used in peak picking and producing hashes
###########

def peak_picking(spect, f_win=5,t_win=5, f_hop_len=20, t_hop_len=20):
    #apply sliding window to spectrogram of size (2*f_win, 2*t_win) to find peaks
    #returns array of dimensionality equal to the spectrogram with 1s at peaks and 0s elsewhere
    peaks = np.zeros(spect.shape)

    f_bins = spect.shape[0]
    hops = spect.shape[1]
    
    f_range =  f_bins - f_win #number of frequency bins strides cover
    h_range = hops - t_win #number of time indices strides cover
    
    #number of strides along each dimension
    #round down number of bins to cover divided by the size the window is shifted by
    f_win_positions = int(np.floor(f_range/f_hop_len))
    t_win_positions = int(np.floor(h_range/t_hop_len))
    
    #slide window across spectrogram
    for f_win_num in range(f_win_positions):
        for t_win_num in range(t_win_positions):
            
            #set index bounds for the window
            f_low = f_win_num*f_hop_len
            f_high = f_low + f_win
            t_low = t_win_num*t_hop_len 
            t_high = t_low + t_win
            
            #get slice of spectrogram
            slice = spect[f_low:f_high, t_low:t_high]

            #find the peak in the window and set it to 1
            peak = np.max(slice)
            slice_res = np.zeros_like(slice)
            slice_res[slice == peak] = 1
            
            #add the peak to the peaks array using window indices
            peaks[f_low:f_high, t_low:t_high] += slice_res
    
    #set all peaks to 1
    peaks[peaks > 0] = 1    
    return peaks


#converts constellation maps to hashes
#hashes are returned as as a list of tuples with (f1,f2,t2-t1)
def get_pair_hash(peaks, t_shift=50, f_win=50, t_win=50):
    peak_cords = np.column_stack(np.where(peaks == 1))
    
    hashes = []
    
    #iterates through peaks and finds all other peaks within a region defined by t_shift, f_win, t_win
    for peak in peak_cords:
        #define window boundaries
        #t_shift is the amount the region is shifted to the right of the anchor
        #f_win is half the frequency window size, t_win is half the time window size
        t_low = peak[1] + t_shift
        t_high = t_low + t_win
        
        f_low = peak[0] - f_win
        f_high = peak[0] + f_win
        
        paired_peaks =  peak_cords[np.where( (peak_cords[:,0] > f_low) & 
                    (peak_cords[:,0] < f_high)& 
                    (peak_cords[:,1] > t_low) & 
                    (peak_cords[:,1] < t_high) 
                    )]

        #create hash tuples
        for pair in paired_peaks:
            #dont add hash for point with itself
            if not (pair[0] == peak[0] and pair[1] == peak[1]):
                hash = (peak[0],pair[0],pair[1]-peak[1]) #hash is a tuple with (f1,f2,t2-t1)
                offset = peak[1]
                hashes.append([hash,offset])
            else:
                continue
        
    return hashes