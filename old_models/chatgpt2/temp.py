import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_story():
    # 1. Load the model and tokenizer from Hugging Face
    # "openai-community/gpt2" is the official repository for the original GPT-2 model
    model_name = "openai-community/gpt2"
    
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 2. Define the starting prompt
    prompt = "A samurai visiting a shrine"

    # 3. Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # 4. Generate the story
    # We use parameters like 'temperature' and 'top_k' to make the story creative
    print("Generating story...")
    output = model.generate(
        input_ids,
        max_length=1000,          # Limit the story length
        num_return_sequences=1,  # Generate only one story
        no_repeat_ngram_size=2,  # Prevent repeating phrases
        do_sample=True,          # Enable sampling for creativity
        top_k=50,                # Limit vocabulary choices (helps coherence)
        top_p=0.95,              # Nucleus sampling
        temperature=0.8,         # Randomness (higher = more creative/chaotic)
        pad_token_id=tokenizer.eos_token_id
    )

    # 5. Decode and print the result
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("-" * 40)
    print(story)
    print("-" * 40)

if __name__ == "__main__":
    generate_story()