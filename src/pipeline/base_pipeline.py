import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader, Dataset


class BasePipeline(pl.LightningModule):
    min_delta = 1e-6
    patience = 20
    max_epochs = 10000

    def __init__(self, ctm, dat, cg_file, ncm):
        super().__init__()
        self.ctm = ctm
        self.dat = dat
        self.cg_file = cg_file
        self.ncm = ncm

    def forward(self, n=1, u=None, do={}):
        return self.ncm(n, u, do)

    def train_dataloader(self):  # 1 epoch = 1 step
        return T.utils.data.DataLoader(
            SCMDataset(self.dat),
            batch_size=4096,
            num_workers=8,
            shuffle=True,
            pin_memory=True
            )
    
class SCMDataset(Dataset):
    def __init__(self, dat_sets):
        # 支持 DataFrame 或 dict
        if hasattr(dat_sets, 'shape'):
            self.length = dat_sets.shape[0]
            self.data = dat_sets
        else:
            # 假设 dat_sets 是 dict
            self.length = len(next(iter(dat_sets.values())))
            self.data = dat_sets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if hasattr(self.data, 'iloc'):
            # DataFrame
            return {k: self.data.iloc[idx][k] for k in self.data.columns}
        else:
            # dict
            return {k: self.data[k][idx] for k in self.data}
