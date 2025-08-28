# generate.py
import torch
import ShakespeareLLM as M  # uses the same model code + tokenizer as training

device = M.device  # keep device consistent with the model's forward()

# rebuild the model exactly as trained (it pulls vocab_size/block_size/etc. from M)
model = M.GPTLanguageModel().to(device)

# load the trained weights
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# prompt -> ids using the same tokenizer from training
prompt = (
    "The history of artificial intelligence dates back to the mid-20th century, "
    "when researchers first began to explore whether machines could simulate human thought. "
    "Early pioneers in the field developed symbolic systems and logic-based programs, "
    "which laid the groundwork for modern machine learning. "
    "By the 1980s, the rise of neural networks introduced new methods of pattern recognition. "
    "In recent years, the field has expanded rapidly due to advances in computational power, "
    "large-scale datasets, and the availability of open-source frameworks. "
    "Today, artificial intelligence is applied in diverse areas such as healthcare, "
    "finance, robotics, and natural language processing. "
    "One of the most important breakthroughs was"
)

context = torch.tensor([M.encode(prompt)], dtype=torch.long, device=device)

# generate
with torch.no_grad():
    out_ids = model.generate(context, max_new_tokens=500, temperature=0.6, top_k=30, top_p=0.9)[0].tolist()

print(M.decode(out_ids))
