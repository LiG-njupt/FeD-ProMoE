#!/usr/bin/python

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
from torch.nn import functional as F

class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def get_triplets(self, labels):
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))

            temp_triplets = [
                [anchor_positive[0], anchor_positive[1], neg_ind]
                for anchor_positive in anchor_positives
                for neg_ind in negative_indices
            ]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

    def forward(self, embeddings, labels):
        triplets = self.get_triplets(labels)

        if len(triplets) != 0:
            triplets = triplets.cuda()

            ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

            losses = F.relu(ap_distances - an_distances + self.margin)
        else:
            losses = torch.zeros(1)

        return losses.mean(), len(triplets)


if __name__ == '__main__':
    label = torch.LongTensor([1, 1, 1, 1, 1, 1])
    loss_fn = TripletLoss(margin=1.0)
    triplets = loss_fn.get_triplets(label)
    print(len(triplets))
