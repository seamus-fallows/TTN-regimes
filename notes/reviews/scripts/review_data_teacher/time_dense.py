import sys, time, yaml, torch
torch.set_num_threads(1)
import run_dense as rd
from run_generic import expand
cfg = yaml.safe_load(open('../../../../../experiments/data_teacher/configs/e4a_sigma.yaml'))
cfg['name'] = 'timing_e4a'
c, sigma, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
case = [k for k in expand(cfg) if k['overlap'] == c][0]
t0 = time.time()
s = rd.job((cfg['name'], case, 'mps', sigma, seed, cfg))
print('TIMING', c, sigma, seed, round(time.time() - t0, 1), s)
