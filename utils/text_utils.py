from transformers import OpenAIGPTTokenizer


# Load the OpenAIGPT tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def clean_text(text):
    # Remove unwanted characters and tokenize the text
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = ' '.join(text.split())
    return text


def encode_text(text):
    # Encode the text using the OpenAIGPT tokenizer
    encoded_text = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=1024)
    return " ".join(encoded_text)
