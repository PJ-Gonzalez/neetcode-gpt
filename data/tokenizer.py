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
        tokens = list(corpus)
        merges = []

        for _ in range(max(0, num_merges)):
            pair_counts = {}

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            highest_frequency = max(pair_counts.values())
            best_pair = min(
                pair
                for pair, count in pair_counts.items()
                if count == highest_frequency
            )
            merges.append(list(best_pair))

            merged_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            tokens = merged_tokens

        return merges
