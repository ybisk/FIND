import numpy as np

###############################################################################
##    This block introduces helper functions
##    to_int and to_string convert AAs back and forth between representations
##    sort_data and pad_data help create batches of data of a fixed length to pass
##      to the network.
###############################################################################

def process(data, acids, lbls):
    strs = np.array([sequence for label, sequence in data])
    inps = [to_int(sequence, acids) for label, sequence in data]
    outs = np.array([lbls[label] for label, sequence in data])
    return strs, inps, outs

def to_int(seq, acids):
    """   Map AA sequence to integers  """
    seq = seq.replace("*","")
    conv = []
    for i in range(len(seq)):
        if seq[i] not in acids:
            print(i, seq)
        conv.append(acids[seq[i]])
    return np.array(conv)

def to_string(seq):
    """  Map ints to AA sequence  """
    return "".join([ints[s] for s in seq])

def sort_data(inputs, outputs, strs=[]):
    """ 
      Sorted by input length and then output length
    """
    if len(strs) == 0:
        strs = [""]*len(inputs)
    v = []
    for i, o, s in zip(inputs, outputs, strs):
        v.append((len(i), i, o, s))
    v.sort(key=lambda x: x[0])

    sorted_inputs = []
    sorted_outputs = []
    sorted_strs= []
    for len_i, i, o, s in v:
        sorted_inputs.append(i)
        sorted_outputs.append(o)
        sorted_strs.append(s)

    return sorted_inputs, sorted_outputs, sorted_strs

def pad_data(inputs):
    max_i = max([len(i) for i in inputs])
  
    padded_i = np.zeros((len(inputs), max_i), dtype=np.int64)
    for i in range(len(inputs)):
        padded_i[i, :len(inputs[i])] = np.copy(inputs[i])

    return padded_i
