import numpy as np
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from data.rec_data_loader import RecDataset


def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    
    return WeightedRandomSampler(weights, len(weights))
    

def create_rec_data_loader(opt, type):
    print('Preparing {} dataset ...'.format(type))
    dataset = RecDataset(opt, type)

    if type == 'train':
        sampler = make_weighted_sampler(dataset.labels)
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size
        )

    return data_loader
    