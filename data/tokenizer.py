from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        
        prev = [c for c in corpus]
        merges = []
        for _ in range(num_merges):
            #print(prev)
            c = Counter((prev[i], prev[i+1] ) for i in range(len(prev)-1))

            new_merge = min(
                c,
                key=lambda x: [-c[x], x]
            )
            merges.append([new_merge[0], new_merge[1]])
            s = "".join(new_merge)
            #print(prev, new_merge)
            curr = list()
            i = 0
            while i < len(prev):
                if i != len(prev)-1 and "".join(prev[i:i+2]) == s:
                    curr.append(s)
                    i += 2
                else:
                    curr.append(prev[i])
                    i += 1
            prev = curr

        return merges