from torch.utils.data.dataset import Dataset
import numpy as np
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask
import torch


class MLMLoader(Dataset):
    def __init__(self, dataframe, token2idx, age2idx, max_len, code='code', age='age'):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.age2idx = age2idx

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code, label = random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)

    def __len__(self):
        return len(self.code)