import sys
import os
import copy

opt = dict()

opt['dataset'] = 'data/OntoNotes'
opt['sparse'] = True
opt['load'] = None
opt['s_load'] = None
opt['lr'] = '1e-3'
opt['decay'] = '1e-3'
opt['t_prepre_epoch'] = 0
opt['t_pre_epoch'] = 1000
opt['s_pre_epoch'] = 400
opt['s_epoch'] = 50
opt['t_epoch'] = 50
opt['seed'] = 1
opt['layer'] = 3
opt['iter'] = 10
opt['min_match'] = 2
opt['uns_scale'] = 1
opt['t_draw'] = 'max'
opt['s_draw'] = 'smp'

opt['device'] = 3
opt['s_device'] = 2

# no_boot default is '', True for alternative
opt['no_boot'] = ''
opt['disable_global'] = True
opt['initialization'] = 'embeeding'
assert opt['initialization'] in ['embedding', 'random', 'uniform']

if opt['layer'] != 3:
    log_path = 'logs/train_onto_layer_%d.log' % opt['layer']
else:
    log_path = 'logs/train_onto.log'

if opt['min_match'] != 2:
    log_path = log_path.replace('.log', '_match%d.log' % opt['min_match'])

if opt['s_draw'] != 'smp':
    log_path = log_path.replace('.log', '_%s.log' % opt['s_draw'])

if opt['seed'] != 1:
    log_path = log_path.replace('.log', '_seed%d.log' % opt['seed'])

if opt['no_boot']:
    log_path = log_path.replace('.log', '_no_boot.log')

if opt['disable_global']:
    log_path = log_path.replace('.log', '_no_global.log')


def generate_command(opt):
    cmd = 'python -u train.py'
    for opt, val in opt.items():
        if val is not None and val != '':
            cmd += ' --' + opt + ' ' + str(val)
    cmd = 'nohup ' + cmd + ' > ' + log_path + ' 2>&1 &'
    return cmd


def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))


run(opt)
