import pandas as pd
import numpy as np
from pathlib import Path
from utils.sequences import encodeDNA
from rich.console import Console
from tqdm import tqdm

from base.base_data_loader import BaseDataLoader


class ConvATACseqDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvATACseqDataLoader, self).__init__(config)
        self.console = Console()
        

    def get_train_data(self):
      
      self.console.print("Loading training data ...")
      X_train, Y_train =  self.loadDataByChrom(chroms = self.config.trainer.train_chroms)

      return X_train, Y_train

    def get_test_data(self):

        self.console.print("Loading test data ...")
        X_test , Y_test = self.loadDataByChrom(chroms = self.config.trainer.test_chroms)

        return X_test, Y_test

    def loadDataByChrom(self, chroms):

        if not Path(self.config.data.path).exists():
            self.console.print(f'Could not find {self.config.data.path}')

        df = pd.read_csv(self.config.data.path,sep="\t")   
        data_touse  = df[ df['chromosome'].isin(chroms)]

        labels = []
        dna_seqs = []

        
        for index, row in tqdm(data_touse.iterrows(), total=data_touse.shape[0]):
            lbl = row.Class
            seq = row.sequences

            if len(seq) != self.config.model.seq_len:
                self.console.print('DNA sequence length is different from "seq_len" specified in the config file.')
            
            dna_seqs.append(seq)
            labels.append(lbl)
    
        self.console.print(f'[bold red]Loaded {len(labels)} sequences.[/bold red]')
        # do one-hot encoding
        self.console.print('Hot-encoding.')
        seqs_hot = encodeDNA(dna_seqs)
        return seqs_hot, np.asanyarray(labels).astype(np.float)
     
