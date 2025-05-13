import itertools
import os
import sys
import warnings

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # isort:skip deterministic PyTorch

import numpy as np  # noqa: E402
import torch as T  # noqa: E402

from src import pipeline  # noqa: E402
from src.run.pipeline import get_key, run  # noqa: E402

import torch
torch.set_float32_matmul_precision('medium')  # 或者 'high'

warnings.filterwarnings(
    "ignore", "Setting attributes on ParameterDict is not supported.")
warnings.filterwarnings("ignore", category=UserWarning,
                        module='pytorch_lightning.utilities.distributed')

T.use_deterministic_algorithms(True)

valid_pipelines = [s for s in dir(pipeline)
                   if s != 'BasePipeline'
                   and isinstance(getattr(pipeline, s), type)
                   and issubclass(getattr(pipeline, s), pipeline.BasePipeline)]

assert sys.argv[1] in valid_pipelines, f'choose from {valid_pipelines}'
assert len(sys.argv) < 3 or sys.argv[2].isdigit()  # gpu to use
assert len(sys.argv) < 4 or sys.argv[3].isdigit()  # sample_size to use

cls = getattr(pipeline, sys.argv[1])
gpu = None if len(sys.argv) < 3 else [int(sys.argv[2])]
#ns = list((10 ** np.linspace(3, 6, 16)).astype(int))[:-1]
ns = [1000]
dims = [4]  # , 2, 4, 8, 16, 32, 64, 128, 256]
n_trials = 1

cg_files = [
    #'dat/cg/napkin.cg',
    'dat/cg/5-ch.cg',
    #'dat/cg/6-cc.cg',
    'dat/cg/9-ch.cg',
    #'dat/cg/15-cc.cg',
    #'dat/cg/17-d.cg',
    #'dat/cg/25-cc.cg',
    #'dat/cg/45-cc.cg',
    'dat/cg/49-ch.cg',
    #'dat/cg/65-d.cg',
    'dat/cg/99-ch.cg',
]

for i, (n, dim, cg_file, trial_index) in enumerate(itertools.product(
        ns, dims, cg_files, range(n_trials))):
    while True:
        try:
            if not run(cls, cg_file, n, dim, trial_index, gpu=gpu, minmax=(sys.argv[1] == "NLLNCMMaxPipeline")):
                break
        except Exception as e:
            print(e)
            print('[failed]', get_key(cg_file, n, dim, trial_index))
            raise
