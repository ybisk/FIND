import time
import gzip
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
###############################################################################
##    Functions for evaluation and visualization
###############################################################################
def run_evaluation(net, validate, 
                   aggregate=False):

    net.train(mode=False)
    net.top_predictor = []
    net.predictors = defaultdict(int)
    """
      Run evaluation
    """
    val_loss = 0.0
    val_acc = []
    gold_counts = np.zeros(net.config.num_labels)
    pred_counts = np.zeros(net.config.num_labels)
    corr_counts = np.zeros(net.config.num_labels)

    if net.config.confusion:
        pairs = np.zeros((net.config.num_labels, net.config.num_labels))

    for datum in  tqdm(validate, ncols=80):
        inps, lens, outs, strs = datum
        inps = inps.to(net.config.device)
        outs = outs.to(net.config.device)
        b_size = len(inps)

        logits, att, full = net(inps)

        val_loss += F.cross_entropy(logits, outs).item()
        _, preds = torch.max(logits, 1)

        preds = preds.data.cpu().numpy()
        outs = outs.data.cpu().numpy()
        val_acc.extend(list((preds == outs)))

        np.add.at(pred_counts, preds, 1)
        np.add.at(gold_counts, outs, 1)
        np.add.at(corr_counts, preds[(preds == outs)], 1)

        if net.config.confusion:
            if not net.config.pconfusion:
                np.add.at(pairs, [outs[vals], preds], 1)
            else:
                dists = F.softmax(logits, -1)
                for i in range(len(vals)):
                    gold = outs[vals][i]

                    tmp = [(dists[i,j], j) for j in range(len(net.config.ilbls))]
                    tmp.sort()
                    prob, second = tmp[-2]
                    pairs[gold, second] += 1

        if aggregate:
            aggregate_predictors(net, strs, outs, full, att)

    if net.config.confusion:
        out = open("confusion.csv", 'w')
        out.write("," + ",".join([net.config.ilbls[i] 
                                  for i in range(len(net.config.ilbls))]) + "\n")
        for i in range(len(net.config.ilbls)):
            out.write("{},".format(net.config.ilbls[i]))
            for j in range(len(net.config.ilbls)):
                out.write("{},".format(pairs[i,j]))
            out.write("\n")
        out.close()
    return val_loss, 100*np.array(val_acc).mean(), (gold_counts, pred_counts, corr_counts)

def aggregate_predictors(net, seqs, outs, full, att):
    dists = F.softmax(full.permute(0, 2, 1), dim=-1)
    if dists.shape[0] == 1:
        att = att.unsqueeze(0)  # batch size of 1 needs to be unsqueezed
    vals = dists * att.unsqueeze(2)
    for b in range(len(seqs)):
        max_val = -1e10
        max_predictor = "NONE"
        max_class = -1
        for i in range(len(att[0])):
            predictor = seqs[b][i:i + net.RF]

            for c in range(net.config.num_labels):
                net.predictors[(predictor, c)] += vals[b,i,c].item() 

            cval, c = torch.max(vals[b,i,:], 0)
            cval = cval.item()
            if cval > max_val:
                max_val = cval
                max_predictor = predictor
                max_class = c.item()
        net.top_predictor.append((max_val, max_predictor, net.config.ilbls[max_class], 
                                  net.config.ilbls[outs[b]], seqs[b].strip()))


def print_predictors(net, epoch):
    rtype = "binary" if net.config.binary is not None else "multi"
    rewht = "reweight" if net.config.reweight else "orig"
    fname = "{}.{}.{}.{}.{}.h{}.b{}".format(net.config.name, epoch, net.RF, 
                                            rtype, rewht, net.config.hidden_dim, 
                                            net.config.batch_size)

    g = gzip.open("{}.predictors.joint.gz".format(fname),'wt')
    joint = defaultdict(list)
    for seq, lbl in net.predictors:
        joint[net.config.ilbls[lbl]].append((net.predictors[(seq, lbl)], seq))

    for lbl in joint:
        vals = joint[lbl]
        vals.sort()
        vals.reverse()
        for val, seq in vals:
            g.write("{:5} {:30} {}\n".format(lbl, seq, val))
        g.write("\n")
    g.close()


    g = gzip.open("{}.top_predictors.txt.gz".format(fname), 'wt')
    g.write("{:10} {:30} {:5} {:5} {}\n".format("Val", "Predictor", "Pred", 
                                                "Gold", "Seq"))
    net.top_predictor.sort()
    net.top_predictor.reverse()
    for val, predictor, pred, gold, seq in net.top_predictor:
        g.write("{:10.9f} {:30} {:5} {:5} {}\n".format(val, predictor, pred, 
                                                       gold, seq))
    g.close()


def print_eval(config, train, test=None): 
    """  Print training performance  """
    gold, pred, corr = train
    p, r = corr / pred, corr / gold
    f = 2*p*r/(p+r)

    if test is not None:
        t_gold, t_pred, t_corr = test
        t_p, t_r = t_corr / t_pred, t_corr / t_gold
        t_f = 2*t_p*t_r/(t_p+t_r)

    gold_counts = [(gold[lab],lab) for lab in range(config.num_labels)]
    gold_counts.sort(reverse=True)
    for count, i in gold_counts:
        train_str = "{:<10} {:<5} {:5.3f} {:5.3f} {:5.3f}   ".format(config.ilbls[i], int(count), p[i], r[i], f[i])
        if test is not None:
            test_str = "{:<5} {:5.3f} {:5.3f} {:5.3f}".format(int(t_gold[i]), t_p[i], t_r[i], t_f[i])
        else:
            test_str = ""
        print(train_str + test_str)
