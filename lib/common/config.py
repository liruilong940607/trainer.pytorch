from yacs.config import CfgNode as CN


_C = CN()

# needed by trainer
_C.name = 'default'
_C.checkpoints_path = '../data/checkpoints/'
_C.results_path = '../data/results/'
_C.learning_rate = 1.0
_C.weight_decay = 0.0
_C.momentum = 0.0
_C.optim = 'Adam'
_C.schedule = [10, 20]
_C.gamma = 0.1
_C.resume = False

# needed by train()
_C.ckpt_path = ''
_C.batch_size = 16
_C.num_threads = 4
_C.num_epoch = 50
_C.freq_plot = 10
_C.freq_save = 1000
_C.freq_eval = 1000

_C.net = CN()
_C.net.backbone = ''

_C.dataset = CN()
_C.dataset.root = ''
_C.dataset.smpl_dir = '/mnt/data/smpl/'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('../configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)
