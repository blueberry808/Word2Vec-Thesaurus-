import torch
import pickle
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F 
from model_def import CBOW_MODEL


model = None
word2idx = None
idx2word = None

def load_everything():
    global model, word2idx, idx2word

    if word2idx is None:
        with open("model_info/word2idx-2.pkl", "rb") as f:
            word2idx = pickle.load(f)

    if idx2word is None:
        with open("model_info/idx2word-2.pkl", "rb") as f:
            idx2word = pickle.load(f)

    if model is None:
        vocab_size = len(word2idx)
        model = CBOW_MODEL(
            vocab_size=vocab_size,
            embedding_size=250,
            context_size=4
        )
        state = torch.load(
            "model_info/cbow_model-4.pt",
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(state)
        model.eval()

    # >>> HERE: return the embedding weights
    embeddings = model.embedding_layer.weight # or model.embeddings.weight if you want the tensor

    return model, word2idx, idx2word, embeddings





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
    
