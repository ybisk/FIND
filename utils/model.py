import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################################################
##    Model definition + Helper Functions
###############################################################################
class Net(nn.Module):
    def __init__(self, config, width=3, RF=19):
        """
           Build a stack of 1D convolutions with batch norm and ReLU activations
           The final two convolutions are simply linear layers, then followed by
           a prediction and attention layer.
        """
        super(Net, self).__init__()
        self.width = width
        self.RF = RF
        self.config = config
    
        self.embedding = nn.Embedding(config.input_dim, config.hidden_dim)
        self.embedding.weight.data[0] = torch.zeros(config.hidden_dim)
        layers = [
          nn.Conv1d(config.hidden_dim, config.hidden_dim, self.width),
          nn.ReLU(),
          nn.BatchNorm1d(config.hidden_dim),
          nn.Conv1d(config.hidden_dim, config.hidden_dim, self.width*2),
          nn.ReLU(),
          nn.BatchNorm1d(config.hidden_dim),
          nn.Conv1d(config.hidden_dim, config.hidden_dim, self.width*4),
          nn.ReLU(),
          nn.BatchNorm1d(config.hidden_dim),
          nn.Conv1d(config.hidden_dim, config.hidden_dim, 1),
          nn.ReLU(),
          nn.Conv1d(config.hidden_dim, config.hidden_dim, 1),
          nn.ReLU(),
        ]

        self.conv_stack = nn.Sequential(*layers)
    
        self.pred = nn.Conv1d(config.hidden_dim, config.num_labels, 1)
        self.att = nn.Conv1d(config.hidden_dim, 1, 1)

    def forward(self, x):
        embed = self.embedding(x).permute(0,2,1)
        embed = self.conv_stack(embed)

        # Log probabilities for every class at every substring
        logits = self.pred(embed)
    
        # Un-normalized weight of a given n-gram
        att = self.att(embed)
        # Reshape [b,L] --> [b,1,L]  -- and normalize
        re_att = F.softmax(att.view(x.size()[0],1,-1), dim=-1)
        # Rescale logits by attention weight
        joint = re_att * logits
        # Class distribution
        collapsed = torch.sum(joint, 2)

        # Turn attention into a distribution
        att = F.softmax(torch.squeeze(att), dim=-1)

        return collapsed, att, logits

    def loss(self, logits, labels, weight):
        if self.config.reweight:
            return F.cross_entropy(logits, labels, weight=weight)
        else:
            return F.cross_entropy(logits, labels)
