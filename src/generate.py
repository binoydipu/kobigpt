import torch
from models import KobiGPTModel, GPTConfig
import pickle
import time

# ---------- Configuration ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "../trained_models/kobigpt_model_2.pt"
max_new_tokens = 500  # how many tokens to generate
temparature = 1.0     # higher temperature = more randomness
# -----------------------------------


# Load utils
with open('../trained_models/kobigpt_utils_2.pkl', 'rb') as f:
    utils = pickle.load(f)
stoi = utils['stoi']
itos = utils['itos']
model_params = utils['params']
config_dict = utils['config']

def encode(s):
        return [stoi[c] if c in stoi else stoi[' '] for c in s]  # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# Load model
config = GPTConfig(**config_dict)
model = KobiGPTModel(config=config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Start token (e.g. zero or custom prompt, 1 is "\n" in our vocab)

# prompt = str("বিনয় ভাবছে আকাশ কেন নীল,")
prompt = str('''বিনয় ভাবছে আকাশের নিচে, বিনয় শুনছে পাখির গান, 
বিনয়ের মনে বাজে নতুন সুর, বিনয় লিখছে কবিতা। 
''')
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
# Alternatively, to start from scratch, uncomment the following line:
# context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate tokens
with torch.no_grad():
    generated = model.generate(
        context, 
        max_new_tokens=max_new_tokens,
        temparature=temparature,
    )

# Save to file
timestr = time.strftime("%Y_%b_%d-%H-%M-%S")
with open(f"../output/poems/generated_poem_{timestr}.txt", "w", encoding="utf-8") as f:
    f.write(f"# Generated Poem\n")
    f.write(f"# Model: KobiGPT Model\n")
    f.write(f"# Model Parameters: {model_params} M\n")
    f.write(f"# Configuration: {config_dict}\n")
    f.write(f"# No. of new Tokens: {max_new_tokens}\n")
    f.write(f"# Temperature: {temparature}\n")
    f.write(f"# Generated on: {timestr}\n\n")
    
    f.write(decode(generated[0].tolist()))
print(f"\nGenerated poem saved to output/poems/generated_poem_{timestr}.txt")
print("\n✨ Generation complete! ✨")