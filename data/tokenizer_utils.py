from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans = []
        for num in numbers:
            l = 0
            n_str = str(num)
            temp = []
            for r in range(len(n_str)):
                if n_str[l:r+1] not in vocab:
                    if l < r:
                        temp.append(n_str[l:r])
                        l = r
                    else:  # l == r: single char not in vocab
                        temp.append(n_str[l:r+1])
                        l = r + 1
            if l != len(n_str):
                temp.append(n_str[l:])
            ans.append(temp)
        return ans

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        l = 0
        while l < len(text):
            best_end = -1
            for r in range(l + 1, len(text) + 1):
                if text[l:r] in vocab:
                    best_end = r          # keep extending to find the longest match
            if best_end == -1:
                best_end = l + 1          # unknown character, consume one
            count += 1
            l = best_end
        return count


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words_count = len(text.split())
        token_count = self.count_tokens(text, vocab)
        return round(token_count / words_count, 4)