import os
from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--prep_path', type=str, default='./datasets/preprocess_styletransfer_classify.npz', help='path to preprocess data')
        parser.add_argument('--real_data_path', type=str, default='./datasets/styletransfer_generate.npz', help='path to real dataset')  # M_gen
        parser.add_argument('--fake_data_path', type=str, default='./datasets/styletransfer_stylized_ours_0.npz', help='path to generated set')  # stylized set
        # parser.add_argument('--fake_data_path', type=str, default='./datasets/styletransfer_stylized_aberman_0.npz', help='path to training set')
        # parser.add_argument('--fake_data_path', type=str, default='./datasets/styletransfer_stylized_holden_0.npz', help='path to training set')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--load_iter', type=int, default=-1)
        parser.add_argument('--load_latest', action='store_true')

        return parser

    def check(self, opt):
        assert opt.model == 'recognition', 'Model should be recognition!'
        assert opt.mode == 'eval', 'Not eval mode!'
