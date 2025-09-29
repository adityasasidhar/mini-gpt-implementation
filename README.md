This repository contains a simple implementation of a Tiny Transformer model, which is a type of neural network used in natural language processing tasks. The project includes:

Transformer Model: A class TinyTransformer that defines the architecture of the transformer model, including attention layers and feedforward networks.

Tokenizer Class: A SimpleTokenizer class to convert text into numerical sequences suitable for training the model.

Training Script: A script to train the Tiny Transformer model on a sample dataset using character-level next-character prediction as the task.

Model Saving/Loading: Functions to save and load the trained model and tokenizer, allowing for easy reuse of the model in different contexts.

Reusability Example: A separate script reuse.py that demonstrates how to load and use the trained model to generate text based on a given starting prompt.

The Main transfoermer component copied from the console log generated using .eval() method:

          TinyTransformer(
            (token_emb): Embedding(81, 64)
            (pos_emb): Embedding(64, 64)
            (layers): ModuleList(
              (0-3): 4 x ModuleDict(
                (attn): TinyAttention(
                  (qkv): Linear(in_features=64, out_features=192, bias=False)
                  (proj): Linear(in_features=64, out_features=64, bias=True)
                )
                (mlp): Sequential(
                  (0): Linear(in_features=64, out_features=128, bias=True)
                  (1): GELU(approximate='none')
                  (2): Linear(in_features=128, out_features=64, bias=True)
                )
                (ln1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                (ln2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              )
            )
            (ln_f): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (head): Linear(in_features=64, out_features=81, bias=False)
          )
