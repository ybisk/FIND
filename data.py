import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

class data:
  def __init__(self, config, test=False):
    ############################################################################
    ## Data is loaded, Train/Validation/Test, counted, converted to numbers and 
    ## stored in numpy arrays.
    ############################################################################
    """ Data & Parameters """
    self.train = [line.strip().split() for line in open(config.training,'r')]
    if test:
      self.valid = [line.strip().split() for line in open(config.testing,'r')]
    else:
      self.valid = [line.strip().split() for line in open(config.validation,'r')]

    # Sanity check formatting of the input data
    for vals in self.train:
        if len(vals) != 2:
            print("Problem: " + vals)
            sys.exit()

    # Build Training Data
    strs, inputs, outputs = self.process(self.train, config.acids, config.lbls)

    print("Training counts\t")
    l_c = defaultdict(int) 
    for v in outputs:
        l_c[config.ilbls[v]] += 1
    V = [(l_c[v],v) for v in l_c]
    V.sort()
    V.reverse()
    print("    ".join(["{}: {}".format(lbl, cnt) for cnt,lbl in V]))

    count = np.zeros(len(config.ilbls), dtype=np.float32)

    for v in range(len(config.ilbls)):
        if config.ilbls[v] in l_c:
            count[v] += l_c[config.ilbls[v]]
    distr = np.sum(count)/(np.size(count)*count) #1. - count/np.sum(count)
    self.weight = torch.from_numpy(100*distr).to(config.device)

    inps, outs, strs = self.sort_data(inputs, outputs, strs)
    outs = np.array(outs)
    strs = np.array(strs)

    # Build Validation Data
    t_strs, t_inps, t_outs = self.process(self.valid, config.acids, config.lbls)
    #t_inps, t_outs, t_strs = self.sort_data(t_inps, t_outs, t_strs)
    #t_outs = np.array(t_outs)
    #t_strs = np.array(t_strs)

    print("Training    Inps: ", len(inputs))
    print("Training    Outs: ", outputs.shape)
    print("Validation  Inps: ", len(t_inps))
    print("Validation  Outs: ", t_outs.shape)
    print("Labels\t", config.lbls)

    self.training = DataLoader(list(zip(inputs, outputs, strs)), shuffle=True)
    self.validate = DataLoader(list(zip(t_inps, t_outs, t_strs)), shuffle=False)


  ##############################################################################
  ## This block introduces helper functions
  ## to_int and to_string convert AAs back and forth between representations
  ## sort_data and pad_data create batches of data of a fixed length to pass
  ## to the network.
  ##############################################################################

  def process(self, data, acids, lbls):
      strs = np.array([sequence for label, sequence in data])
      inps = [self.to_int(sequence, acids) for label, sequence in data]
      outs = np.array([lbls[label] for label, sequence in data])
      return strs, inps, outs

  def to_int(self, seq, acids):
      """   Map AA sequence to integers  """
      seq = seq.replace("*","")
      conv = []
      for i in range(len(seq)):
          if seq[i] not in acids:
              print(i, seq)
          conv.append(acids[seq[i]])
      return np.array(conv)

  def to_string(self, seq):
      """  Map ints to AA sequence  """
      return "".join([ints[s] for s in seq])

  def sort_data(self, inputs, outputs, strs=[]):
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

  def pad_data(self, inputs):
      max_i = max([len(i) for i in inputs])
    
      padded_i = np.zeros((len(inputs), max_i), dtype=np.int64)
      for i in range(len(inputs)):
          padded_i[i, :len(inputs[i])] = np.copy(inputs[i])

      return padded_i

