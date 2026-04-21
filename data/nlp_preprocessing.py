import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sents = positive + negative
        
        vocab = {w: i+1 for i,w in enumerate(sorted(set(w for sent in all_sents for w in sent.split())))}

        encoded = [torch.tensor([vocab[w] for w in sent.split()]) for sent in all_sents]

        ans = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)

        return ans