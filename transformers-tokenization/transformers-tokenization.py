import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        ind = 3

        for text in texts:
            for word in text.split(" "):
                word_idx = self.word_to_id.get(word, None)
                if word_idx is None:
                    self.word_to_id[word] = ind + 1
                    ind += 1
        
        for k,v in self.word_to_id.items():
            self.id_to_word[v] = k
        
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        result = []
        for word in text.split(" "):
            result.append(self.word_to_id.get(word, 1))
        return result
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        result = []
        for word_id in ids:
            result.append(self.id_to_word.get(word_id, self.unk_token))
        return " ".join(result)
