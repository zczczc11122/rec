import os
import random
import torch
import numpy as np

def set_fixed_seed(seed, args, checkpoint):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # 想随机种子的结果完全一致 torch.backends.cudnn.deterministic 要设置成 False，不过训练速度会下降
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    if args.resume:
        #这里主要是保证断点重训和一直不间断训练结果一模一样
        if checkpoint is not None:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['py_rng_state'])

