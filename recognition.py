import time
import torch

from options.rec_options import RecOptions
from data import create_rec_data_loader
from data.rec_data_loader import InputFetcher
from model import create_model
from processor_recognition import Processor


if __name__ == '__main__':
    rec_options = RecOptions()
    opt = rec_options.parse()
    print('Start training on cuda:%s' % opt.gpu_ids)

    # create data loader
    train_loader = create_rec_data_loader(opt, 'train')
    test_loader = create_rec_data_loader(opt, 'test')
    train_size = len(train_loader)
    test_size = len(test_loader)
    train_fetcher = InputFetcher(opt, train_loader)
    test_fetcher = InputFetcher(opt, test_loader)

    model = create_model(opt)    
    trainer = Processor(opt)

    if opt.load_latest:
        model.load_networks()
        opt.resume_iter = model.get_current_iter()
    elif opt.resume_iter > 0:
        model.load_networks(opt.resume_iter)

    # train!
    start_time = time.time()
    for iter in range(opt.resume_iter, opt.total_iters):
        inputs = next(train_fetcher)
        data = inputs['x']['posrot'][:, :3]  # use only position info
        data = torch.unsqueeze(data, dim=-1)

        if opt.criteria == 'content':
            label = inputs['c']  # [N]
        elif opt.criteria == 'style':
            label = inputs['s']  # [N]

        out, loss = trainer.forward(model, data, label)

        model.reset_grad()
        loss.backward()
        model.optimizer.step()

        # print loss
        if (iter + 1) % opt.print_every == 0:
            print('[Iteration {}/{}] Training Loss: {:.5f}'.format(iter + 1, opt.total_iters, loss))

        # evaluate
        if (iter + 1) % opt.eval_every == 0:
            test_loss, top1_acc = trainer.evaluate(model, test_fetcher, test_size)
            print('[Iteration {}/{}] Test Loss: {:.5f} | Top1 Acc: {:.2f}'.format(iter + 1, opt.total_iters, test_loss, 100 * top1_acc))

        # save the latest model
        if (iter + 1) % opt.save_latest_every == 0:
            model.save_networks(iter + 1, latest=True)
        
        # save the model
        if (iter + 1) % opt.save_every == 0:
            model.save_networks(iter + 1)
            