import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# ==========================================
# 1. Core Implementation: Manual Layer Norm
# ==========================================

class CustomLayerNorm(nn.Module):
    """
    Implementation of Layer Normalization from scratch.
    Paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        # normalized_shape is usually the dimension of the features (hidden size)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: gamma (gain) and beta (bias)
        # Shape matches the feature dimension
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x shape: [Batch_Size, ..., Features]
        
        # 1. Calculate Mean and Variance across the FEATURE dimension (last dim)
        # Unlike Batch Norm, which calculates over the Batch dimension (dim 0),
        # Layer Norm calculates stats for EACH sample independently.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 2. Normalize
        # x_hat = (x - mean) / sqrt(var + epsilon)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 3. Scale and Shift
        # y = gamma * x_hat + beta
        output = x_norm * self.gamma + self.beta
        
        return output

# ==========================================
# 2. Advanced Application: LN-LSTM Cell
# ==========================================

class LayerNormLSTMCell(nn.Module):
    """
    A manual LSTM cell incorporating Layer Normalization.
    LN is particularly beneficial for RNNs as it stabilizes hidden state dynamics.
    """
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear transformations for Input (x) and Hidden (h)
        # We stack the gates (input, forget, cell, output) for efficiency
        # Shapes: input -> 4 * hidden, hidden -> 4 * hidden
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # Layer Normalization layers for the gates
        # The paper suggests normalizing the summed inputs before the nonlinearity.
        self.ln_ih = CustomLayerNorm(4 * hidden_size)
        self.ln_hh = CustomLayerNorm(4 * hidden_size)
        
        # Cell state normalization (optional but often used in LN-LSTM)
        self.ln_c = CustomLayerNorm(hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state

        # 1. Compute linear projections
        # Apply LN separately to input-to-hidden and hidden-to-hidden transformations
        gates_ih = self.ln_ih(self.weight_ih(x))
        gates_hh = self.ln_hh(self.weight_hh(h_prev))

        # 2. Combine
        gates = gates_ih + gates_hh

        # 3. Split into specific gates (Input, Forget, Cell, Output)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # 4. Apply activations
        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        # 5. Update Cell State
        c_t = (f_t * c_prev) + (i_t * g_t)
        
        # Apply Layer Norm to the cell state (common variant in LN-LSTM papers)
        c_t_norm = self.ln_c(c_t)

        # 6. Update Hidden State
        h_t = o_t * torch.tanh(c_t_norm)

        return h_t, c_t

class SequenceModel(nn.Module):
    """
    Row-by-row MNIST Classifier using the LN-LSTM Cell.
    Treats image as sequence of 28 rows of 28 pixels.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LayerNormLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [Batch, Sequence_Len, Features] -> [Batch, 28, 28]
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Iterate through sequence (time steps)
        for t in range(seq_len):
            x_t = x[:, t, :] # select row t
            h_t, c_t = self.cell(x_t, (h_t, c_t))

        # Classification on final hidden state
        out = self.fc(h_t)
        return out

# ==========================================
# 3. Training and Evaluation on MNIST
# ==========================================

def main():
    # Configuration
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.002
    EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {DEVICE}")

    # Data Strategy: Row-Sequential MNIST
    # We treat the image as a sequence to demonstrate RNN capabilities.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading/Loading MNIST...")
    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    # Input size 28 (width of image), Hidden size 128, Classes 10
    model = SequenceModel(input_size=28, hidden_size=HIDDEN_SIZE, num_classes=10).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    print("\nStarting Training with Layer Normalized LSTM...")
    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # Reshape images: [Batch, 1, 28, 28] -> [Batch, 28, 28]
            images = images.squeeze(1).to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping is standard for RNNs, but LN helps mitigate the need.
            # We leave it unclipped here to show LN stability, or use a high value.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Complete. Avg Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%. Time: {time.time()-start_time:.1f}s")

    # Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.squeeze(1).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy of the LayerNorm-LSTM on MNIST: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()