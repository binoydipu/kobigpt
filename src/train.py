import torch
from models import KobiGPTModel, GPTConfig
from tqdm import tqdm
import pickle
import time


# ------------ Hyperparameters ------------
batch_size = 32                                         # no of independent sequences processed in parallel
block_size = 256                                       # context length
max_iters = 8000                                       # no of steps to train
eval_interval = 500                                     # interval to evaluate the loss
learning_rate = 3e-4                                    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 100                                        # no of iterations to estimate the loss
grad_clip = 1.0
# --------------------------------------------

torch.set_float32_matmul_precision('medium')

model_path = "../trained_models/kobigpt_model.pt"

print(f"Using device: {device}")
print(f"Training for {max_iters} steps with batch size {batch_size} and block size {block_size}")


def load_dataset():
    # load the dataset
    with open('../dataset/tagore_poems.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # create a set of all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    # create mappings from characters to integers and vice versa
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # encode the entire text dataset and convert to a tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # split the data into training and validation sets (90% train, 10% val)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return vocab_size, train_data, val_data, decode, stoi, itos

def get_batch(data):
    # generate a random batch of data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(train_data if split == 'train' else val_data)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- Scheduler Function ---
def get_lr(step):
    pass


vocab_size, train_data, val_data, decode, stoi, itos = load_dataset()


# ----------- Model Setup -----------
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True
)

model = KobiGPTModel(config=config)
model = model.to(device)
model_params = sum(p.numel() for p in model.parameters())/1e6
print(f'Model params: {model_params} M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ----------- Logging Setup -----------
timestr = time.strftime("%Y_%b_%d-%H-%M-%S")
with open(f"../output/logs/training_log_{timestr}.txt", "w") as log_file:
    log_file.write(f"Training KobiGPT Model\n")
    log_file.write(f"Dataset: Tagore Poems\n")
    log_file.write(f"Model Parameters: {model_params} M\n")
    log_file.write(f"Configuration: {config.__dict__}\n\n")
    log_file.write(f"Hyperparameters:\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Block Size: {block_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Max Iters: {max_iters}\n")
    log_file.write(f"Eval Interval: {eval_interval}\n\n")
    log_file.write(f"Starting training...\n\n")
    log_file.write(f"Step\tTrain Loss\tVal Loss\n")


# ----------- Training Loop -----------
for step in tqdm(range(max_iters)):

    if step % eval_interval == 0:
        losses = estimate_loss(model, train_data, val_data)
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open(f"../output/logs/training_log_{timestr}.txt", "a") as log_file:
            log_file.write(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")
    
    # to save model checkpoints
    if step % 1000 == 0:
        torch.save(model.state_dict(), f"../trained_models/checkpoints/kobigpt_step_{step}.pt")

    # sample a batch of data
    xb, yb = get_batch(train_data)
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient clipping
    optimizer.step()


# --- Save the final model ---
torch.save(model.state_dict(), model_path)

# --- Save utils for generation ---
with open('../trained_models/kobigpt_utils.pkl', 'wb') as f:
    pickle.dump({
        'stoi': stoi, 
        'itos': itos, 
        'params': model_params, 
        'config': config.__dict__,
    }, f)

print(f"Training complete. Model saved to {model_path}")
print("You can now run 'python src/generate.py' to generate text using the trained model.")
