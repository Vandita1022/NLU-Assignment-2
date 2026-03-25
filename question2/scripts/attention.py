import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F

# =====================
# 1. DATA PREP (Same)
# =====================
with open("TrainingNames.txt", "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

chars = sorted(list(set("".join(names))))
chars = ['<PAD>', '<START>', '<END>'] + chars
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

def encode(name):
    return [stoi['<START>']] + [stoi[c] for c in name] + [stoi['<END>']]

X, Y = [], []
for name in names:
    seq = encode(name)
    X.append(torch.tensor(seq[:-1]))
    Y.append(torch.tensor(seq[1:]))

X_padded = pad_sequence(X, batch_first=True, padding_value=stoi['<PAD>'])
Y_padded = pad_sequence(Y, batch_first=True, padding_value=stoi['<PAD>'])

class NameDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

loader = DataLoader(NameDataset(X_padded, Y_padded), batch_size=32, shuffle=True)

# =====================
# 2. MANUAL ATTENTION MODEL
# =====================
class ManualAttentionGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # --- Manual GRU Weights ---
        # Update Gate (z), Reset Gate (r), Candidate (n)
        self.W_ih = nn.Parameter(torch.randn(embed_size, hidden_size * 3) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size * 3) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size * 3))

        # --- Attention Weights (Bahdanau Style) ---
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        # --- Output Layer ---
        # Concatenates current hidden + attention context
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def gru_step(self, x_t, h_prev):
        # Calculate gates
        gates = x_t @ self.W_ih + h_prev @ self.W_hh + self.b
        i_z, i_r, i_n = gates.chunk(3, dim=1)
        
        z_t = torch.sigmoid(i_z)
        r_t = torch.sigmoid(i_r)
        n_t = torch.tanh(i_n * r_t) # candidate hidden state
        
        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t

    def forward(self, x):
        batch_size, seq_len = x.size()
        embeds = self.embedding(x)
        
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        history = []
        logits_list = []

        for t in range(seq_len):
            # 1. Update Hidden State
            h_t = self.gru_step(embeds[:, t, :], h_t)
            
            # 2. Causal Attention
            if len(history) > 0:
                # Look back at all previous hidden states
                prev_states = torch.stack(history, dim=1) # [batch, t, hidden]
                
                # Score = v * tanh(W * prev_states)
                scores = self.v(torch.tanh(self.attn_linear(prev_states))) # [batch, t, 1]
                weights = torch.softmax(scores, dim=1)
                
                # Context = Weighted sum of previous states
                context = torch.sum(weights * prev_states, dim=1) # [batch, hidden]
            else:
                context = torch.zeros_like(h_t)

            # 3. Predict next char
            combined = torch.cat([h_t, context], dim=1) # [batch, hidden*2]
            logits = self.fc(combined)
            logits_list.append(logits.unsqueeze(1))
            
            history.append(h_t)

        return torch.cat(logits_list, dim=1)

# =====================
# 3. TRAINING
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ManualAttentionGRU(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])

print("Training Manual Attention Model...")
for epoch in range(200):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =====================
# 4. GENERATION (With Temperature & Top-K)
# =====================
def generate_attentional(model, temp=1.1, k=5):
    model.eval()
    with torch.no_grad():
        # Start with <START> and a random first letter to boost variety
        valid_starts = list(stoi.keys())[3:]
        first_char = random.choice(valid_starts)
        input_indices = [stoi['<START>'], stoi[first_char]]
        
        result = [first_char]
        
        for _ in range(15):
            x = torch.tensor([input_indices]).to(device)
            logits = model(x)[:, -1, :] / temp # take last prediction
            
            # Block unwanted tokens
            logits[0, stoi['<PAD>']] = -float('inf')
            logits[0, stoi['<START>']] = -float('inf')
            
            probs = torch.softmax(logits, dim=1)
            
            # Top-K Sampling
            top_v, top_i = torch.topk(probs, k)
            idx = top_i[0, torch.multinomial(top_v, 1)].item()
            
            if idx == stoi['<END>']: break
            
            char = itos[idx]
            result.append(char)
            input_indices.append(idx)
            
        return "".join(result)

print("\n--- Attention Generated Names ---")
generated = [generate_attentional(model) for _ in range(10)]
for n in generated: print(n)

# --- EVALUATION EXPORT ---
print("\nGenerating 100 names for evaluation...")
with open("attention_generated.txt", "w") as f:
    for _ in range(100):
        name = generate_attentional(model) # Ensure this matches your function name
        f.write(name + "\n")
print("Saved to attention_generated.txt")