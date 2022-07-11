import os
import argparse

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='recog', choices=['recog', 'eval'])
        parser.add_argument('--model', type=str, default='recognition')
        parser.add_argument('--output_dir', type=str, default='./output')
        parser.add_argument('--experiment_name', type=str, default='experiment_name')  # specify experiment name
        parser.add_argument('--criteria', type=str, default='content', choices=['content', 'style'])  # criteria: CRA or SRA
        parser.add_argument('--styles', type=str, nargs='+', default=styles)
        parser.add_argument('--contents', type=str, nargs='+', default=contents)
        parser.add_argument('--num_styles', type=int, default=len(styles))
        parser.add_argument('--num_contents', type=int, default=len(contents))
        parser.add_argument('--clip_size', type=int, nargs='+', default=[32, 21])
        parser.add_argument('--gpu_ids', type=str, default='0')

        self.initialized = True

        return parser

    def gather_options(self):
        parser = None
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        self.parser = parser

        return parser.parse_args()

    def check(self, opt):
        pass

    def parse(self):
        opt = self.gather_options()
        opt.save_dir = make_dir(opt.output_dir, opt.experiment_name)
        self.check(opt)

        return opt
        

def make_dir(parent_dir, dir_name):
    child_dir = os.path.join(parent_dir, dir_name)
    if not os.path.exists(child_dir):
        os.makedirs(child_dir)
        
    return child_dir
    