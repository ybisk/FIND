class config:
  name       = "BD"                    # prefix for models/logs
  labels     = "data/bd.labels.txt"    # Classes
  training   = "data/bd.labeled.train" # file location
  validation = "data/bd.labeled.val"   # file location
  testing    = "data/bd.labeled.test"  # file location
  batch_size = 32                      # Batch size
  epochs     = 100                     # Number of training epochs
  hidden_dim = 256                     # Hidden dim
  lr         = 1e-3                    # learning rate
  binary     = None                    # Binary classifer based on argument label
  reweight   = False                   # Balance class distribution
  test       = False                   # Swap validation for test set
  load       = None                    # Load model
  log        = "./logs/"               # Directory for logs
  confusion  = False                   # If test, generate confusion matrix
  pconfusion = False                   # sum probabilities for confusion
