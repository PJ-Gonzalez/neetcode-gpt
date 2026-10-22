from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._greedy_tokenize(str(number), vocab) for number in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        token_count = self.count_tokens(text, vocab)
        return round(token_count / word_count, 4)
        
    # A shared helper keeps the greedy matching logic consistent 
    # It tries longer vocabulary entries first, then falls back 
    # to a single character if none match.
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        max_token_length = max((len(token) for token in vocab), default=1)
        tokens = []
        i = 0

        while i < len(text):
            match = None
            max_length = min(max_token_length, len(text) - i)

            for length in range(max_length, 0, -1):
                candidate = text[i : i + length]
                if candidate in vocab:
                    match = candidate
                    break

            if match is None:
                match = text[i]

            tokens.append(match)
            i += len(match)

        return tokens
