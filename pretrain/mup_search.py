import numpy as np
import csv
import subprocess
import re
import os
os.environ["WANDB_MODE"] = "disabled"
# Number of experiments
N = 10
# List to store results
results = []

# You can modify the config path as needed
TRAIN_SCRIPT = 'cs336-basics/scripts/train.py'
CONFIG_NAME = 'experiment/mup_search'

for i in range(N):
    # Sample parameters
    embeddings_scale = np.random.uniform(1, 20)
    init_std = np.exp(np.random.uniform(np.log(1e-4), np.log(0.2)))
    lr = np.exp(np.random.uniform(np.log(1e-4), np.log(0.1)))
    embeddings_scale = float(f"{embeddings_scale:.2e}")
    init_std = float(f"{init_std:.2e}")
    lr = float(f"{lr:.2e}")

    # Build the command
    cmd = [
        'uv','run',
        'python', TRAIN_SCRIPT,
        f'--config-name={CONFIG_NAME}',
        f'training.embeddings_scale={embeddings_scale}',
        f'training.init_std={init_std}',
        f'training.lr={lr}'
    ]
    print(f"Running experiment {i+1}: {' '.join(cmd)}")

    # Run train.py and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + '\n' + result.stderr

    # Try to extract train loss and valid loss
    # Adjust the regex according to the output format of train.py if needed
    train_loss = None
    valid_loss = None
    train_match = re.search(r"Training step [0-9]+, Loss: ([0-9\.eE+-]+)", output)
    valid_match = re.search(r"Final estimated validation loss: ([0-9\.eE+-]+)", output)
    if train_match:
        train_loss = float(train_match.group(1))
    if valid_match:
        valid_loss = float(valid_match.group(1))

    results.append({
        'embeddings_scale': embeddings_scale,
        'init_std': init_std,
        'lr': lr,
        'train_loss': train_loss,
        'valid_loss': valid_loss
    })

    print(f"Experiment {i+1} finished: train_loss={train_loss}, valid_loss={valid_loss}")

# Save results to csv
csv_path = os.path.join(os.path.dirname(__file__), 'search_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"All experiments finished. Results saved to {csv_path}")
