from fastapi import FastAPI
from load_model import load_everything
from load_model import closest_words   # your search function
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F 
from model_def import CBOW_MODEL


app = FastAPI()



# Load once at startup
model, word2idx, idx2word, embeddings = load_everything()

print(idx2word)
print("Embedding layer:", model.embedding_layer.weight)

@app.get("/similar")
def get_similar(word: str, top_k: int = 10):
    return closest_words(model, word, word2idx, idx2word, top_k)





'''
def closest_words(model, word, word_to_idx, idx_to_word, top_k=10):
    # 1. look up index of the word
    if word not in word_to_idx:
        return f"'{word}' not in vocabulary"

    word_idx = word_to_idx[word]

    # 2. get embedding for the query word (1 x d)
    word_vec = model.embedding_layer.weight[word_idx]  # (d,)
    word_vec = word_vec.unsqueeze(0)                   # (1, d)

    # 3. get all embeddings (vocab_size x d)
    all_embeds = model.embedding_layer.weight          # (V, d)

    # 4. compute cosine similarity with every word
    sims = F.cosine_similarity(word_vec, all_embeds)   # (V,)

    # 5. sort by similarity (descending)
    topk = torch.topk(sims, top_k + 1)                 # +1 to skip the word itself

    # 6. convert indices back to words
    results = []
    for idx in topk.indices:
        w = idx_to_word[int(idx)]
        if w != word:   # skip the query word
            results.append(w)

        if len(results) == top_k:
            break

    return results
    


'''
