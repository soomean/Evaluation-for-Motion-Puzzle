import os
from .base_options import BaseOptions


class RecOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--prep_path', type=str, default='./datasets/preprocess_styletransfer_classify.npz', help='path to preprocess data')
        parser.add_argument('--train_data_path', type=str, default='./datasets/styletransfer_classify.npz', help='path to training dataset')  # M_cls
        parser.add_argument('--test_data_path', type=str, default='./datasets/styletransfer_generate.npz', help='path to test dataset')  # M_gen
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--resume_iter', type=int, default=0)
        parser.add_argument('--total_iters', type=int, default=3000)
        parser.add_argument('--eval_every', type=float, default=100)
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--beta1', type=float, default=0.99)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--print_every', type=int, default=10)
        parser.add_argument('--save_latest_every', type=int, default=100)
        parser.add_argument('--save_every', type=int, default=100)
        parser.add_argument('--load_latest', action='store_true')

        return parser

    def check(self, opt):
        assert opt.model == 'recognition', 'Model should be recognition!'
        assert opt.mode == 'recog', 'Not recog mode!'
