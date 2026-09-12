import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
text = "I am not happy with this product."

tokens = word_tokenize(text)

print(tokens)