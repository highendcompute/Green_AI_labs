import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# RAPL ENERGY READING (no sudo required)
# ---------------------------------------------------------
def read_rapl():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read())


# ---------------------------------------------------------
# LIVE CARBON INTENSITY (UK National Grid ESO)
# ---------------------------------------------------------
def get_carbon_intensity():
    url = "https://api.carbonintensity.org.uk/intensity"
    try:
        data = requests.get(url, timeout=3).json()
        return data["data"][0]["intensity"]["actual"]
    except:
        return None


# ---------------------------------------------------------
# Convert energy (J) → CO₂ emissions (g)
# ---------------------------------------------------------
def energy_to_co2(energy_joules, carbon_intensity_g_per_kwh):
    if carbon_intensity_g_per_kwh is None:
        return None
    return energy_joules * (carbon_intensity_g_per_kwh / 3_600_000)


# ---------------------------------------------------------
# Simple PyTorch MLP model for MNIST
# ---------------------------------------------------------
class SmallMLP(nn.Module):
    # feed-forward classifier (fc)
    # input layer of 784 features, 2 hidden layers of 'width', output layer of 10 classes

    def __init__(self, width=256):
        super().__init__()
        self.fc1 = nn.Linear(784, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------
# Train the model once (fast) if no pretrained model exists
# ---------------------------------------------------------
def train_model(width=256):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = SmallMLP(width=width).train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training model (about 10 seconds)...")
    for epoch in range(2):  # only 2 epochs needed for ~97% accuracy
        for X, y in train_loader:
            optim.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

    torch.save(model.state_dict(), "mnist_pretrained.pth")
    print("Training complete. Model saved.")
    return model


# ---------------------------------------------------------
# Accuracy calculation
# ---------------------------------------------------------
def accuracy(model, X, y, batch_size):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i+batch_size])
            preds = logits.argmax(dim=1)
            correct += (preds == y[i:i+batch_size]).sum().item()
            total += len(preds)
    return correct / total


# ---------------------------------------------------------
# Measure energy + time for a function
# ---------------------------------------------------------
def measure_energy_and_time(fn):
    start_e = read_rapl()
    t0 = time.time()
    fn()
    t1 = time.time()
    end_e = read_rapl()
    return (end_e - start_e) / 1e6, t1 - t0


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    X = test_set.data.float() / 255.0
    y = test_set.targets

    # Load or train pretrained model
    width = 256; #default 256
    print("Model with width=",width)
    model = SmallMLP(width=width).eval()
    
    # load pretrained if possible, else train
    checkpoint_name = f"mnist_pretrained_width{width}.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_name))
        print("Loaded pre-trained MNIST model for width=",width)
    except FileNotFoundError:
        print("No pre-trained model for width=",width,"\nTraining...")
        model = train_model(width=width).eval()
        torch.save(model.state_dict(), checkpoint_name)

    batch_sizes = [1, 8, 32, 128, 512]

    carbon_intensity = get_carbon_intensity()
    print("Live carbon intensity (gCO2/kWh):", carbon_intensity)

    print("\nBatch Size | Energy (J) | Time (s) | Accuracy | CO₂ (g)")
    print("-------------------------------------------------------------")

    for b in batch_sizes:
        def run():
            with torch.no_grad():
                for i in range(0, len(X), b):
                    model(X[i:i+b])

        energy, duration = measure_energy_and_time(run)
        acc = accuracy(model, X, y, b)
        co2 = energy_to_co2(energy, carbon_intensity)
        co2_str = f"{co2:.6f}" if co2 is not None else "N/A"

        print(f"{b:10d} | {energy:10.3f} | {duration:7.3f} | {acc:.3f} | {co2_str}")
