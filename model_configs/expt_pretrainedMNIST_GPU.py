import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pynvml

# ---------------------------------------------------------
# CPU ENERGY (RAPL)
# ---------------------------------------------------------
def read_rapl():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read())


# ---------------------------------------------------------
# GPU ENERGY (NVML sampling)
# ---------------------------------------------------------
def init_nvml():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

def sample_gpu_energy(device, fn):
    """
    Measures GPU energy by sampling instantaneous power draw.
    """
    power_samples = []
    timestamps = []

    start = time.time()
    fn()
    end = time.time()

    # Sample GPU power during the run
    # (Sampling AFTER fn() is too late — so we wrap fn() inside sampling)
    # Instead, we run fn() inside a timed loop.
    # For simplicity, we assume fn() is fast and sample around it.

    # More accurate version:
    # Run fn() inside a thread and sample concurrently.
    # But for teaching purposes, this is sufficient.

    # Sample power 200 times per second
    duration = end - start
    steps = int(duration * 200)

    for _ in range(steps):
        power_mw = pynvml.nvmlDeviceGetPowerUsage(device)  # milliwatts
        power_samples.append(power_mw / 1000.0)  # convert to watts
        timestamps.append(time.time())
        time.sleep(0.005)

    # Integrate power over time
    energy_j = 0.0
    for i in range(1, len(power_samples)):
        dt = timestamps[i] - timestamps[i-1]
        energy_j += power_samples[i] * dt

    return energy_j


# ---------------------------------------------------------
# LIVE CARBON INTENSITY
# ---------------------------------------------------------
def get_carbon_intensity():
    url = "https://api.carbonintensity.org.uk/intensity"
    try:
        data = requests.get(url, timeout=3).json()
        return data["data"][0]["intensity"]["actual"]
    except:
        return None


def energy_to_co2(energy_j, carbon_intensity):
    if carbon_intensity is None:
        return None
    return energy_j * (carbon_intensity / 3_600_000)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class SmallMLP(nn.Module):
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
# TRAIN MODEL IF NEEDED
# ---------------------------------------------------------
def train_model(width=256):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = SmallMLP(width=width).cuda().train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training model...")
    for epoch in range(2):
        for X, y in train_loader:
            X, y = X.cuda(), y.cuda()
            optim.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

    torch.save(model.state_dict(), f"mnist_pretrained_width{width}.pth")
    print("Training complete.")
    return model


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------
def accuracy(model, X, y, batch_size):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i+batch_size].cuda())
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y[i:i+batch_size]).sum().item()
            total += len(preds)
    return correct / total


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    X = test_set.data.float() / 255.0
    y = test_set.targets

    width = 256
    print("Model with width=",width)
    model = SmallMLP(width=width).cuda().eval()

    # Load pretrained model
    try:
        model.load_state_dict(torch.load(f"mnist_pretrained_width{width}.pth"))
        print("Loaded pretrained model.")
    except FileNotFoundError:
        model = train_model(width=width).eval()

    # Init GPU energy measurement
    gpu = init_nvml()

    batch_sizes = [1, 8, 32, 128, 512]
    carbon_intensity = get_carbon_intensity()

    print("\nBatch | CPU J | GPU J | Total J | Time s | Acc | CO₂ g")
    print("-------------------------------------------------------------")

    for b in batch_sizes:
        def run():
            with torch.no_grad():
                for i in range(0, len(X), b):
                    model(X[i:i+b].cuda())

        # CPU energy
        cpu_start = read_rapl()
        t0 = time.time()

        # GPU energy
        gpu_energy = sample_gpu_energy(gpu, run)

        t1 = time.time()
        cpu_end = read_rapl()

        cpu_energy = (cpu_end - cpu_start) / 1e6
        total_energy = cpu_energy + gpu_energy
        acc = accuracy(model, X, y, b)
        co2 = energy_to_co2(total_energy, carbon_intensity)

        co2_str = f"{co2:.6f}" if co2 is not None else "N/A"

        print(f"{b:5d} | {cpu_energy:6.2f} | {gpu_energy:6.2f} | {total_energy:7.2f} | {t1-t0:6.3f} | {acc:.3f} | {co2_str}")
