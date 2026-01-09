import torch
import numpy as np
from collections import Counter
import re
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

class IMDBProcessor:
    def __init__(self, max_vocab_size = 25000, max_seq_len = 200):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.vocab_built = False
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9/s]', ' ', text)
        return word_tokenize(text)

    def build_vocab(self, texts):

        all_words = []
        for text in texts:
            all_words.extend(self.tokenize(text))

        word_counts = Counter(all_words)
        
        sorted_words = sorted(word_counts, key = word_counts.get, reverse = True)

        for word in sorted_words[:self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built. Size: {len(self.word2idx)}")
        print(list(self.word2idx.items())[:10])
        self.vocab_built = True

    def process_text(self, text):
        if not self.vocab_built:
            raise ValueError("Vocabulary must be built before processing text.")
        
        tokens = self.tokenize(text)
        
        indices = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        indices = indices[:self.max_seq_len]
        padding = [self.word2idx["<pad>"] for _ in range(self.max_seq_len - len(indices))]
        indices.extend(padding)
        return torch.tensor(indices)

    def load_pretrained_embeddings(self, embedding_dim = 100):

        w2v_model = api.load(f"glove-wiki-gigaword-{embedding_dim}")
        embedding_matrix = np.zeros((len(self.word2idx), embedding_dim))

        hits = 0
        misses = 0

        for word,idx in self.word2idx.items():
            if idx < 2: continue 
            
            try:
                embedding_vector = w2v_model[word]
                embedding_matrix[idx] = embedding_vector
                hits += 1
            except KeyError:
                # If word not in Word2Vec, leave as zeros (or random init)
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
                misses += 1

        print(f"Pretrained embeddings loaded. Hits: {hits}, Misses: {misses}")
        return torch.tensor(embedding_matrix, dtype=torch.float)


class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.data = hf_dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        
        input_idxs = self.processor.process_text(text)
        
        return input_idxs, torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":
    # Dummy data
    dummy_reviews = [
        "This movie was fantastic", 
        "I hated this film it was awful",
        "The plot was good but the acting was bad"
    ]
    
    processor = IMDBProcessor()
    processor.build_vocab(dummy_reviews)
    
    vector = processor.process_text("This movie was fantastic")
    print("\nSample Vector:", vector)
    
    weights = processor.load_pretrained_embeddings()
    print("Embedding Matrix Shape:", weights.shape)
    print(weights[processor.word2idx["fantastic"]])