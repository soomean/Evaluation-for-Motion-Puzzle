import os
import numpy as np

from options.eval_options import EvalOptions
from data import create_rec_data_loader
from data.rec_data_loader import InputFetcher
from model import create_model
from processor_recognition import Processor
from utils.metrics import calculate_activations_labels, calculate_activations_statistics, calculate_fid


if __name__ == '__main__':
    test_options = EvalOptions()
    opt = test_options.parse()
    print('Start evaluation on cuda:%s' % opt.gpu_ids)

    # create data loader
    real_loader = create_rec_data_loader(opt, 'real')
    fake_loader = create_rec_data_loader(opt, 'fake')
    real_size = len(real_loader)
    fake_size = len(fake_loader)
    print('The number of real data = %d' % real_size)
    print('The number of fake data = %d' % fake_size)
    real_fetcher = InputFetcher(opt, real_loader)
    fake_fetcher = InputFetcher(opt, fake_loader)

    model = create_model(opt)
    tester = Processor(opt)

    if opt.load_latest:
        model.load_networks()
        opt.load_iter = model.get_current_iter()
    else:
        model.load_networks(opt.load_iter)
    print('Parameters are loaded from the iteration %d' % opt.load_iter)

    # calculate accuracy
    _, top1_acc = tester.evaluate(model, fake_fetcher, fake_size)
    acc_type = 'CRA' if opt.criteria == 'content' else 'SRA'
    print('[Evaluation] {}: {:.2f}'.format(acc_type, 100 * top1_acc))
    
    # calculate FMD
    if opt.criteria == 'content':
        gt_features, gt_labels = calculate_activations_labels(model, real_fetcher, real_size, criteria=opt.criteria)
        gt_statistics = calculate_activations_statistics(gt_features)

        features, labels = calculate_activations_labels(model, fake_fetcher, fake_size, criteria=opt.criteria)
        statistics = calculate_activations_statistics(features)

        fid = calculate_fid(gt_statistics, statistics)

        print('[Evaluation] FMD: {:.5f}'.format(fid))
        