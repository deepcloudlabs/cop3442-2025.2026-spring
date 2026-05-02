import torch
from transformers import AutoModel, AutoTokenizer

#MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_ID = "distilbert-base-uncased"
SENTENCE = "He ate it all"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()
inputs = tokenizer(SENTENCE, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
with torch.no_grad():
    embedding_layer = model.get_input_embeddings()
    token_embeddings = embedding_layer(input_ids)
for token, embedding in zip(tokens, token_embeddings[0]):
    print(f"Token: {token}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}\n")
