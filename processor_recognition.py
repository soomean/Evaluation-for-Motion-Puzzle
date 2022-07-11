import numpy as np
import torch
import torch.nn as nn
from model.recognition_model import RecognitionModel


class Processor:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.loss = nn.CrossEntropyLoss()

    def forward(self, model: RecognitionModel, data, label):
        out = model.recognizer(data)
        loss = self.loss(out, label)

        return out, loss

    def extract_feature(self, model: RecognitionModel, data):
        out, feature = model.recognizer.extract_feature(data)

        return out, feature

    def test(self, model: RecognitionModel, data, label):
        model.eval()
        with torch.no_grad():
            out = model.recognizer(data)
        loss = self.loss(out, label)
        
        return out, loss

    def evaluate(self, model: RecognitionModel, test_fetcher, test_size):
        model.eval()
        total_loss = 0
        result_frag = []
        label_frag = []
        for i in range(test_size):
            inputs = next(test_fetcher)
            data = inputs['x']['posrot'][:, :3]  # [N, 3, 32, 21]
            data = torch.unsqueeze(data, dim=-1)
            if self.opt.criteria == 'content':
                label = inputs['c']  # [N]
            elif self.opt.criteria == 'style':
                label = inputs['s']  # [N]

            with torch.no_grad():
                out = model.recognizer(data)
            loss = self.loss(out, label)
            total_loss += loss.item()

            result_frag.append(out.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            
        results = np.concatenate(result_frag)
        labels = np.concatenate(label_frag)
        total_loss /= test_size
        
        for k in [1, 3]:
            if k == 1:
                top1_acc = show_topk(k, results, labels)
        
        return total_loss, top1_acc


def show_topk(k, results, labels):
    rank = results.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(labels)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    return accuracy
    