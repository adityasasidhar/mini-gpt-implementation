import pickle
import math
import torch
import torch.nn as nn
from transformer import *

context_len = 64
batch_size = 16
vocab_size = 81

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loaded_model = TinyTransformer(vocab_size=vocab_size, d_model=64, n_layers=4, context_len=context_len).to(device)
loaded_model.load_state_dict(torch.load("tiny_transformer.pt"))
loaded_model.eval()
with open("tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)


# Generate text
def generate(model, tokenizer, start_text, length=50):
    model.eval()
    context = tokenizer.encode(start_text)[-context_len:]
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    generated = list(context[0].cpu().numpy())
    for _ in range(length):
        inp = torch.tensor([generated[-context_len:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(inp)
        next_token = torch.argmax(logits[0, -1], dim=-1).item()
        generated.append(next_token)
    return tokenizer.decode(generated)


print("Generated:", generate(loaded_model, loaded_tokenizer, "A ", length=20))
