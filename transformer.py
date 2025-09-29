import torch
import torch.nn as nn
import math


class TinyAttention(nn.Module):
    def __init__(self, d_model, context_len):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Attention weights
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)
        att = att.masked_fill(self.mask[:t, :t] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)

        output = att @ v
        return self.proj(output)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=2048, d_model=64, n_layers=4, context_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': TinyAttention(d_model, context_len),
                'mlp': 
                    nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Linear(2 * d_model, d_model)
                ),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)

        x = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            x = x + layer['attn'](layer['ln1'](x))
            x = x + layer['mlp'](layer['ln2'](x))

        x = self.ln_f(x)
        return self.head(x)


class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[tok] for tok in tokens])


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with open("data.txt", 'r', encoding='utf-8') as f:
        data = f.read()

    tokenizer = SimpleTokenizer(data)
    vocab_size = tokenizer.vocab_size

    context_len = 64
    batch_size = 16

    def get_batch():
        ix = torch.randint(len(data) - context_len, (batch_size,))
        x = torch.stack([torch.tensor(tokenizer.encode(data[i:i + context_len])) for i in ix])
        y = torch.stack([torch.tensor(tokenizer.encode(data[i + 1:i + context_len + 1])) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


    # Model
    model = TinyTransformer(vocab_size=vocab_size, d_model=64, n_layers=4, context_len=context_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        model.train()
        inputs, targets = get_batch()

        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


    # Save model and tokenizer
    torch.save(model.state_dict(), "tiny_transformer.pt")
    import pickle

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)