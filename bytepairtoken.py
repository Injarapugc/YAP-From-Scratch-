from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with 'Ġ'
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including 'Ġ' if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]

        # Extend unique_chars with characters from processed_text that are not already included
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        # Optionally, ensure 'Ġ' is included if it is relevant to your text processing
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

        # Now create the vocab and inverse vocab dictionaries
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:  # No more pairs to merge. Stopping training.
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
    def load_vocab_and_merges_form_openai(self,vocab_path,bpe_merges_path):
        with open(vocab_path,'r',encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab={int(v):k for k,v in loaded_vocab.items()}
            self.inverse_vocab={k:int(v) for k,v in loaded_vocab.items()}
            with open(bpe_merges_path,'w',encoding="utf-8") as file:
                lines=file.readlines()
                if lines and lines[0].startswith("#"):
                    lines=lines[1:]
                for rank,line in enumerate(lines):
                    pair=tuple(line.strip().split())
                    if len(pair) !=2:
                        print(f"line {rank +1} has more than 2 entries {line.strip()}")
                        continue
                    token1,token2=pair
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        tokenid1=self.inverse_vocab[token1]
                        tokenid2=self.inverse_vocab[token2]
                        merged_token=token1+token2
                        if merged_token in self.inverse_vocab:
                            merged_token_id=self.inverse_vocab[merged_token]
                            self.bpe_merges[(tokenid1,tokenid2)] = merged_token_id
                        else:
                            print(f"Merged token '{merged_token}' not found in vocab. Skipping.")
                    else:
                        print(f"Skipping pair {pair} as one of the tokens is not in the vocabulary.")



