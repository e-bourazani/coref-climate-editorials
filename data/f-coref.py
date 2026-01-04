from fastcoref import FCoref
import torch
from transformers import AutoTokenizer

# Load model
model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')
from fastcoref import FCoref
import torch

model = FCoref(device='cuda' if torch.cuda.is_available() else 'cpu')

text = "Alice went home. She was tired."
preds = model.predict([text])

print(preds)
# Sample text
text = """Alice went to the store. She bought milk. 
The cashier thanked her before Alice left."""

# Run coreference
preds = model.predict(texts=[text])

print(preds)
