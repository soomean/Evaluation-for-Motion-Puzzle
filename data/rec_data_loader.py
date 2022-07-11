import numpy as np
import torch
from torch.utils import data

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]


class RecDataset(data.Dataset):
    def __init__(self, opt, type='train'):
        if type == 'train':
            data_path = opt.train_data_path
        elif type == 'test':
            data_path = opt.test_data_path
        elif type == 'real':
            data_path = opt.real_data_path
        elif type == 'fake':
            data_path = opt.fake_data_path
        else:
            raise NotImplementedError('Not implemented dataset type!')
        
        self.clips = np.load(data_path)['clips']
        self.feet = np.load(data_path, allow_pickle=True)['feet']
        self.classes = np.load(data_path)['classes']

        self.preprocess = np.load(opt.prep_path)
        self.samples, self.contacts, self.targets, self.labels = self.make_dataset(opt)

    def make_dataset(self, opt):
        X, F, C, S = [], [], [], []
        for dom in range(opt.num_contents):
            dom_idx = [ci for ci in range(len(self.classes))
                       if self.classes[ci][0] == contents.index(opt.contents[dom])]  # index list that belongs to the domain
            dom_clips = [self.clips[cli] for cli in dom_idx]  # clips list (motion data) that belongs to the domain
            dom_feet = [self.feet[fti] for fti in dom_idx]
            dom_styles = [self.classes[si][1] for si in dom_idx]
            X += dom_clips
            F += dom_feet
            C += [dom] * len(dom_clips)
            S += dom_styles

        return X, F, C, S

    def __getitem__(self, index):
        x = self.samples[index]
        f = self.contacts[index]
        x = normalize(x, self.preprocess['Xmean'], self.preprocess['Xstd'])
        data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}
        c = self.targets[index]
        s = self.labels[index]

        return {'x': data, 'c': c, 's': s}

    def __len__(self):
        return len(self.labels)


class InputFetcher:
    def __init__(self, opt, loader):
        self.loader = loader
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    def fetch_src(self):
        try:
            src = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            src = next(self.iter)
            
        return src

    def __next__(self):
        inputs = {}
        src = self.fetch_src()
        inputs_src = {'x': src['x'], 'c': src['c'], 's': src['s']}
        inputs.update(inputs_src)

        return to(inputs, self.device)


def normalize(x, mean, std):
    x = (x - mean) / std

    return x
    

def denormalize(x, mean, std):
    x = x * std + mean

    return x


def to(inputs, device, expand_dim=False):
    for name, ele in inputs.items():
        if isinstance(ele, dict):
            for k, v in ele.items():
                if expand_dim:
                    v = torch.unsqueeze(torch.tensor(v), dim=0)
                ele[k] = v.to(device, dtype=torch.float)
        else:
            if expand_dim:
                ele = torch.unsqueeze(torch.tensor(ele), dim=0)
            if name.startswith('c') or name.startswith('s'):
                inputs[name] = ele.to(device, dtype=torch.long)
            else:
                inputs[name] = ele.to(device, dtype=torch.float)
                
    return inputs
    