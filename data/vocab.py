from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        sorted_uniq = sorted(set(text))
        return (
            {c: i for i,c in enumerate(sorted_uniq)},
            {i: c for i,c in enumerate(sorted_uniq)},
        )

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        return [
            stoi[c]
            for c in text
        ]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        return "".join(
            [itos[i] for i in ids]
        )
