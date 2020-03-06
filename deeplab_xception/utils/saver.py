import os
import time
import shutil
import os.path as osp

import torch


class Saver(object):

    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.directory = osp.join('run', opt.dataset)
        self.experiment_name = time.strftime("%Y%m%d_%H%M%S") + '_' + mode
        self.experiment_dir = osp.join(self.directory, self.experiment_name)
        self.logfile = osp.join(self.experiment_dir, 'experiment.log')
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        for key, val in self.opt._state_dict().items():
            line = key + ': ' + str(val)
            self.save_experiment_log(line)

    def save_checkpoint(self, state, is_best, filename='checkpoint.path.tar'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(osp.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('epoch {}: {}'.format(state['epoch'], best_pred))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))

    def save_experiment_log(self, line):
        with open(self.logfile, 'a') as f:
            f.write(line + '\n')

    def save_eval_result(self, stats):
        with open(os.path.join(self.experiment_dir, 'result.txt'), 'a') as f:
            f.writelines(stats + '\n')

    def backup_result(self):
        backup_root = osp.join(osp.expanduser('~'), "Cache")
        if not osp.exists(backup_root):
            os.mkdir(backup_root)
        backup_dir = osp.join(backup_root, self.experiment_name)
        assert not osp.exists(backup_dir), "experiment has already backup"
        os.mkdir(backup_dir)
        for file in os.listdir(self.experiment_dir):
            source_file = osp.join(self.experiment_dir, file)
            if osp.isfile(source_file):
                shutil.copy(source_file, backup_dir)
