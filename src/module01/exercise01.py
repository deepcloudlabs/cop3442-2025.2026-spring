import tiktoken

text = "aaabdaaabac"

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print(f"Input text: {text}")
print(f"# of tokens: {len(tokens)}")
print(f"Token IDs: {tokens}")