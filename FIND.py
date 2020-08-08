"""
   This section of the code simply sets up all possible variables we might want to change during training.
"""
import sys
import random
import numpy as np                     # Math and Deep Learning libraries
import torch                
from tqdm import tqdm                  # Pretty status bars
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="HCO", help="HCO, NITRO, BD")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--lr", type=float)
parser.add_argument("--evaluate", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--model", type=str)
args = parser.parse_args()

np.seterr(divide='ignore')             # Ignore divide by zero errors
np.warnings.filterwarnings('ignore')
torch.cuda.manual_seed(20180119)       # <-- set a value for consistency

# Load network architecture
from utils.model import Net

# Helper functions for running evaluation and visualization
from utils.analysis import *           

# Choose from configs.[bd, hco, nitro] or make your own
from config import config
config = config(args)

# Data
from data import data
data = data(config, test=args.test)

# Model
net = Net(config=config)
net.to(config.device)
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)


if not args.evaluate:
  #############################################################################
  ##    Training Loop
  #############################################################################

  
  gold_counts = np.zeros(config.num_labels)
  pred_counts = np.zeros(config.num_labels)
  corr_counts = np.zeros(config.num_labels)

  prev_acc = 0
  prev_loss = 1e100
  for epoch in range(0, config.epochs + 1):
      total_loss = 0.0
      train_acc = []

      for datum in tqdm(data.training, ncols=80):
          inps, lens, outs, _ = datum
          # Setup
          optimizer.zero_grad()
          inputs = inps.to(config.device)
          labels = outs.to(config.device)
  
          # Predict
          net.train(mode=True)
          logits, att, full = net(inputs)
          ce_loss = net.loss(logits, labels, weight=data.weight)
  
          # Compute loss and update
          loss = ce_loss
          total_loss += ce_loss.item()
          loss.backward()
          optimizer.step()
      
          # Look at predictions
          _, preds = torch.max(logits, 1)
          dists = full.permute(0,2,1).cpu().data.numpy()
      
          preds = preds.data.cpu().numpy()

          np.add.at(pred_counts, preds, 1)
          np.add.at(gold_counts, outs, 1)
          np.add.at(corr_counts, preds[preds == outs.numpy()], 1)
  
          train_acc.extend(list(preds == outs.numpy()))
    
      # Evaluate on validation (during training)
      val_loss, val_acc, val_counts = run_evaluation(net, data.validate)
      if epoch % 10 == 0 or epoch == config.epochs:
          print_eval(net.config, (gold_counts, pred_counts, corr_counts), val_counts)
  
      print("Epoch: {}  Train Loss: {:8.4f}  Acc: {:5.2f}  Val  Loss {:8.4f}  Acc: {:5.2f}".format(epoch, 
            total_loss, 100*np.array(train_acc).mean(), val_loss, val_acc))
  
      # Save best validation model for optimal generalization
      if val_acc > prev_acc:
          prev_acc = val_acc
          pref = config.name
          torch.save(net, "{}.model".format(config.run_name))
      
      if abs(prev_loss - total_loss)/total_loss < 0.01:
          print("Converged")
          break
      prev_loss = total_loss

else:
  # ## Evaluation and Analysis
  config.load = args.model
  net = torch.load(config.load)                             # Load Saved Model
  net.to(config.device)

  # Run evaluation with best validation model on test
  loss, acc, _ = run_evaluation(net, data.validate)
  print("Acc: {:5.3f}".format(acc))
