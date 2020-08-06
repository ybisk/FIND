import torch

class config:
  def __init__(self, args):
    if args.type == "HCO":
      self._hco() 
    elif args.type == "NITRO":
      self._nitro() 
    elif args.type == "BD":
      self._bd() 
    else:
      raise Exception("Invalid category: {}".format(category))
      
    self.binary     = None                       # Binary classifer based on argument'
    self.reweight   = False                      # Balance class distribution
    self.test       = False                      # Swap validation for test set
    self.load       = None                       # Load model
    self.confusion  = False                      # If test, generate confusion matrix
    self.pconfusion = False                      # sum probabilities for confusion
    self.log        = "./logs/"                  # Directory for logs
    self.epochs     = args.epochs                # Number of training epochs
    if args.hidden_dim is not None:
      self.hidden_dim = args.hidden_dim          # Hidden dim
    if args.batch_size is not None:
      self.batch_size = args.batch_size          # Batch size
    if args.lr is not None:
      self.lr = args.lr                          # Learning rate
    

    # Dictionary of acids and labels
    self.load_acids_and_labels()

    # Handle Binary classification and convert labels to ints
    self.create_labels()

    # Use a GPU when possible
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def _hco(self):
    self.name       = "hco"                      # prefix for models/logs
    self.labels     = "data/hco.labels.txt"      # Classes
    self.training   = "data/hco.labeled.train"   # file location
    self.validation = "data/hco.labeled.val"     # file location
    self.testing    = "data/hco.labeled.test"    # file location
    self.lr         = 5e-3                       # learning rate

  def _nitro(self):
    self.name       = "nitro"                    # prefix for models/logs
    self.labels     = "data/nitro.labels.txt"    # Classes
    self.training   = "data/nitro.labeled.train" # file location
    self.validation = "data/nitro.labeled.val"   # file location
    self.testing    = "data/nitro.labeled.test"  # file location
    self.lr         = 5e-3                       # learning rate

  def _bd(self):
    self.name       = "BD"                       # prefix for models/logs
    self.labels     = "data/bd.labels.txt"       # Classes
    self.training   = "data/bd.labeled.train"    # file location
    self.validation = "data/bd.labeled.val"      # file location
    self.testing    = "data/bd.labeled.test"     # file location
    self.lr         = 1e-3                       # learning rate


  def pretty(self):
    run = str(int(time.time()))
    run += "_binary" if self.binary is not None else "_multi"
    run += "_{}_e{}_h{}_b{}".format(self.name, self.epochs, 
                                    self.hidden_dim, self.batch_size)
    if self.reweight:
        run += "_reweight"
    return run

  def load_acids_and_labels(self):
    ############################################################################
    ## Infrastructure to process Data into Numpy Arrays of Integers
    ## We convert letters to numbers via the acids dictionary.
    ## Also compute the set of function labels (optionally training as binary)
    ############################################################################
    acids = {' ':0, 'A':1, 'C':2, 'E':3, 'D':4, 'G':5, 'F':6, 'I':7, 'H':8,
             'K':9, 'M':10, 'L':11, 'N':12, 'Q':13, 'P':14, 'S':15, 'R':16,
             'T':17, 'W':18, 'V':19, 'Y':20, 'X':21 }
    self.input_dim = len(acids)
    self.ints = {}
    for v in acids:
        self.ints[acids[v]] = v

    L = [line.strip() for line in open(self.labels,'r')]

    # For logging we will store details of our training regime in the file name
    self.run_name = self.pretty()
    self.create_labels()


  def create_labels(self):
    ## Data Preparation
    ############################################################################
    ## If we are training a binary classifier, we need to resplit the data 
    ## into chosen label vs OTHER
    ############################################################################
    lbls = {}    # Map label to integer
    ilbls = {}   # Map integer back to label

    # Perform Multiclass Classification
    if self.binary is None:
        for v in L:
            lbls[v] = len(lbls)
            ilbls[lbls[v]] = v
        self.num_labels = len(lbls)
    else:
        # Split dataset into one-vs-all
        for v in L:
            if v == self.binary:
                lbls[v] = 1
                ilbls[lbls[v]] = v
            else:
                lbls[v] = 0
                ilbls[lbls[v]] = "OTHER"
        print(lbls, ilbls)
        self.num_labels = 2
        run_name += "_" + self.binary

    self.lbls = lbls
    self.ilbls = ilbls
