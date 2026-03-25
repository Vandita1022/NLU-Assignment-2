import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. DATA LOADING
# =========================
with open("TrainingNames.txt", "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

chars = sorted(list(set("".join(names))))
vocab = ['<PAD>', '<SOS>', '<EOS>'] + chars

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)

def encode(name):
    return [stoi['<SOS>']] + [stoi[c] for c in name] + [stoi['<EOS>']]

# =========================
# 2. DATASET
# =========================
X, Y = [], []
for name in names:
    seq = encode(name)
    X.append(torch.tensor(seq[:-1]))
    Y.append(torch.tensor(seq[1:]))

X_pad = pad_sequence(X, batch_first=True, padding_value=stoi['<PAD>'])
Y_pad = pad_sequence(Y, batch_first=True, padding_value=stoi['<PAD>'])

class NameDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

loader = DataLoader(NameDataset(X_pad, Y_pad), batch_size=64, shuffle=True)

# =========================
# 3. LSTM CELL (SCRATCH)
# =========================
class LSTMCellScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

# =========================
# 4. FAST BLSTM
# =========================
class FastBLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.fwd_cell = LSTMCellScratch(hidden_size, hidden_size)
        self.bwd_cell = LSTMCellScratch(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        x = self.embedding(x)

        h_f = torch.zeros(batch_size, self.hidden_size).to(device)
        c_f = torch.zeros(batch_size, self.hidden_size).to(device)

        h_b = torch.zeros(batch_size, self.hidden_size).to(device)
        c_b = torch.zeros(batch_size, self.hidden_size).to(device)

        fwd_outputs = []
        bwd_outputs = [None] * seq_len

        # Forward
        for t in range(seq_len):
            h_f, c_f = self.fwd_cell(x[:, t, :], h_f, c_f)
            fwd_outputs.append(h_f)

        # Backward
        for t in reversed(range(seq_len)):
            h_b, c_b = self.bwd_cell(x[:, t, :], h_b, c_b)
            bwd_outputs[t] = h_b

        outputs = []
        for t in range(seq_len):
            h = torch.cat([fwd_outputs[t], bwd_outputs[t]], dim=1)
            outputs.append(self.fc(h).unsqueeze(1))

        return torch.cat(outputs, dim=1)

# =========================
# 5. TRAINING
# =========================
model = FastBLSTM(vocab_size, 128).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("Training FAST BLSTM...")

for epoch in range(30):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =========================
# 6. GENERATION (FIXED PAD BUG)
# =========================
def generate_name(model, max_len=15):
    model.eval()
    with torch.no_grad():
        idx = stoi['<SOS>']
        seq = [idx]

        for _ in range(max_len):
            x = torch.tensor([seq]).to(device)
            out = model(x)

            logits = out[0, -1]

            # 🚫 BLOCK BAD TOKENS
            logits[stoi['<PAD>']] = -1e9
            logits[stoi['<SOS>']] = -1e9

            probs = torch.softmax(logits, dim=0)
            idx = torch.multinomial(probs, 1).item()

            if idx == stoi['<EOS>']:
                break

            seq.append(idx)

        return "".join([itos[i] for i in seq[1:]])

# =========================
# 7. SAMPLE OUTPUT
# =========================
print("\nGenerated Names:")
for _ in range(10):
    print(generate_name(model))

# =========================
# 8. SAVE OUTPUT
# =========================
with open("blstm_generated.txt", "w") as f:
    for _ in range(100):
        f.write(generate_name(model) + "\n")

print("✅ Saved 100 names to blstm_generated.txt")