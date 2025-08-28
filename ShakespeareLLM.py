import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

# -----------------
# Config
# -----------------
batch_size = 16
block_size = 256
max_iters = 50000
eval_interval = 500
learning_rate = 3e-4
warmup_iters = 1000
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 100
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.1
checkpoint_path = "checkpoint.pt"

torch.manual_seed(1337)

# -----------------
# Load dataset
# -----------------

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

train_text = "\n".join(dataset["train"]["text"])
val_text   = "\n".join(dataset["validation"]["text"])
test_text  = "\n".join(dataset["test"]["text"])

with open("train.txt", "w", encoding="utf-8") as f:
    f.write(train_text)

with open("val.txt", "w", encoding="utf-8") as f:
    f.write(val_text)

with open("test.txt", "w", encoding="utf-8") as f:
    f.write(test_text)

print("WikiText-103 prepared: train.txt, val.txt, test.txt created.")

with open("train.txt", "r", encoding="utf-8") as f:
    train_text = f.read()
with open("val.txt", "r", encoding="utf-8") as f:
    val_text = f.read()

chars = sorted(list(set(train_text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] if i in itos else "?" for i in l])

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

print(f"Vocab size: {vocab_size}, Train length: {len(train_data)}, Val length: {len(val_data)}")

def get_batch(split):
    d = train_data if split=="train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------
# Model
# -----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -block_size:]

            # forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last step logits

            # apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k filtering
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("Inf")

            # top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = -float("Inf")
                logits.scatter_(1, sorted_indices, sorted_logits)

            # sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # append to sequence
            idx = torch.cat((idx, next_id), dim=1)

        return idx


    #def generate(self,idx,max_new_tokens):
     #   for _ in range(max_new_tokens):
      #      idx_cond = idx[:,-block_size:]
       #     with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #        logits,_ = self(idx_cond)
         #   logits = logits[:,-1,:]
          #  probs = F.softmax(logits,dim=-1)
           # idx_next = torch.multinomial(probs, num_samples=1)
            #idx = torch.cat((idx,idx_next),dim=1)
        #return idx

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------
# LR Scheduler
# -----------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return 0.0
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


#main
if __name__ == "__main__":



    # -----------------
    # Checkpoint resume
    # -----------------

    start_iter = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"] + 1
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        steps = checkpoint.get("steps", [])
        print(f"Resumed from iter {start_iter} with loss {checkpoint['loss']:.4f}")
    else:
        train_losses, val_losses, steps = [], [], []

    # -----------------
    # Training loop (with tqdm + live plot)
    # -----------------

    train_losses, val_losses = [], []
    steps = []

    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Train Loss")
    line2, = ax.plot([], [], label="Val Loss")
    ax.legend()
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    plt.show()

    for iter in tqdm(range(start_iter, max_iters+1), desc="Training", unit="step"):

        xb,yb = get_batch("train")

        # update learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb,yb)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        #scaler.scale(loss).backward()

        # gradient clipping
        #scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #scaler.step(optimizer)
        #scaler.update()

        # log
        if iter % eval_interval == 0:
            losses = estimate_loss()
            steps.append(iter)
            train_losses.append(losses["train"].item())
            val_losses.append(losses["val"].item())

            # update live plot
            line1.set_xdata(steps)
            line1.set_ydata(train_losses)
            line2.set_xdata(steps)
            line2.set_ydata(val_losses)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")

            # save checkpoint
            torch.save({
                "iter": iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": losses['val'].item(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "steps": steps
            }, checkpoint_path)

    # -----------------
    # Generate sample
    # -----------------
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
