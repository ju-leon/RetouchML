"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
See: https://github.com/kristpapadopoulos/keras-stochastic-weight-averaging
"""
import os
import glob
import pickle
import argparse
from dnnlib.tflib import init_tf

filepath = 'output.pkl'

def fetch_models_from_files(model_list):
    for fn in model_list:
        with open(fn, 'rb') as f:
            yield pickle.load(f)

def apply_swa_to_checkpoints(models):
    gen, dis, gs = next(models)
    print('Loading', end='', flush=True)
    mod_gen = gen
    mod_dis = dis
    mod_gs = gs
    epoch = 0
    try:
        while True:
            epoch += 1
            gen, dis, gs = next(models)
            if gs is None:
                print("")
                break
            mod_gen.apply_swa(gen, epoch)
            mod_dis.apply_swa(dis, epoch)
            mod_gs.apply_swa(gs, epoch)
            print('.', end='', flush=True)
    except:
        print("")
    return (mod_gen, mod_dis, mod_gs)


parser = argparse.ArgumentParser(description='Perform stochastic weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('results_dir', help='Directory with network checkpoints for weight averaging')
parser.add_argument('--filespec', default='network*.pkl', help='The files to average')
parser.add_argument('--output_model', default='network_avg.pkl', help='The averaged model to output')
parser.add_argument('--count', default=6, help='Average the last n checkpoints', type=int)

args, other_args = parser.parse_known_args()
swa_epochs = args.count
filepath = args.output_model
files = glob.glob(os.path.join(args.results_dir,args.filespec))
if (len(files)>swa_epochs):
    files = files[-swa_epochs:]
files.sort()
print(files)
init_tf()
models = fetch_models_from_files(files)
swa_models = apply_swa_to_checkpoints(models)

print('Final model parameters set to stochastic weight average.')
with open(filepath, 'wb') as f:
    pickle.dump(swa_models, f)
print('Final stochastic averaged weights saved to file.')
