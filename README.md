This repository contains a simple implementation of a Tiny Transformer model, which is a type of neural network used in natural language processing tasks. The project includes:

Transformer Model: A class TinyTransformer that defines the architecture of the transformer model, including attention layers and feedforward networks.

Tokenizer Class: A SimpleTokenizer class to convert text into numerical sequences suitable for training the model.

Training Script: A script to train the Tiny Transformer model on a sample dataset using character-level next-character prediction as the task.

Model Saving/Loading: Functions to save and load the trained model and tokenizer, allowing for easy reuse of the model in different contexts.

Reusability Example: A separate script reuse.py that demonstrates how to load and use the trained model to generate text based on a given starting prompt.
