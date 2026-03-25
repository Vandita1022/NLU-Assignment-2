import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

# =====================
# 1. DATA LOADING & VOCAB
# =====================
with open("TrainingNames.txt", "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

# Include PAD, START, and END tokens
chars = sorted(list(set("".join(names))))
chars = ['<PAD>', '<START>', '<END>'] + chars

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(name):
    return [stoi['<START>']] + [stoi[c] for c in name] + [stoi['<END>']]

# =====================
# 2. DATASET & LOADER
# =====================
X, Y = [], []
for name in names:
    seq = encode(name)
    X.append(torch.tensor(seq[:-1]))  # Predict next char
    Y.append(torch.tensor(seq[1:]))

X_padded = pad_sequence(X, batch_first=True, padding_value=stoi['<PAD>'])
Y_padded = pad_sequence(Y, batch_first=True, padding_value=stoi['<PAD>'])

class NameDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

loader = DataLoader(NameDataset(X_padded, Y_padded), batch_size=32, shuffle=True)

# =====================
# 3. MANUAL RNN FROM SCRATCH
# =====================
class ManualVanillaRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN Weights (Manual Initialization)
        # Input to Hidden
        self.W_ih = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        
        # Hidden to Hidden
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        
        # Hidden to Output
        self.W_ho = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, h=None):
        batch_size, seq_len = x.size()
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Embed the input
        embeds = self.embedding(x) # [batch, seq, hidden]
        
        outputs = []
        
        # Manual Recurrence Loop
        for t in range(seq_len):
            x_t = embeds[:, t, :] # current input [batch, hidden]
            
            # RNN Math: h_t = tanh(x_t*W_ih + h_prev*W_hh + bias)
            h = torch.tanh(x_t @ self.W_ih + self.b_ih + h @ self.W_hh + self.b_hh)
            
            # Output: y_t = h_t*W_ho + b_o
            y_t = h @ self.W_ho + self.b_o
            outputs.append(y_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), h

# =====================
# 4. TRAINING
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ManualVanillaRNN(vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

print(f"Training Manual RNN with {sum(p.numel() for p in model.parameters())} parameters...")

for epoch in range(200): # Running 100 epochs for brevity
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =====================
# 5. GENERATION
# =====================
def generate_name(model, max_len=15):
    model.eval()
    with torch.no_grad():
        char_idx = stoi['<START>']
        hidden = None
        result = []
        
        for _ in range(max_len):
            input_tensor = torch.tensor([[char_idx]]).to(device)
            output, hidden = model(input_tensor, hidden)
            
            probs = torch.softmax(output[0, -1], dim=0)
            char_idx = torch.multinomial(probs, 1).item()
            
            if char_idx == stoi['<END>']:
                break
            
            result.append(itos[char_idx])
            
        return "".join(result)

print("\n--- Manual RNN Generated Names ---")
for _ in range(10):
    print(generate_name(model))

# --- EVALUATION EXPORT ---
print("\nGenerating 100 names for evaluation...")
with open("vanilla_generated.txt", "w") as f:
    for _ in range(100):
        name = generate_name(model) # Ensure this matches your function name
        f.write(name + "\n")
print("Saved to vanilla_generated.txt")