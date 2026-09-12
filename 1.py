import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize

text = "I am learning NLP and building an AI Agent."

tokens = word_tokenize(text)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)