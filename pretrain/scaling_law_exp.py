import numpy as np
import csv
import subprocess
import re
import os
# from scalinglaw import ChinchillaScalingExperiment

os.environ["WANDB_MODE"] = "disabled"

# Training script and config
TRAIN_SCRIPT = 'cs336-basics/scripts/train.py'
CONFIG_NAME = 'experiment/scaling_law'

# Manually defined experiment parameter list (optional)
MANUAL_EXPERIMENTS = [
    # Format: [d_model, d_ff, num_heads, num_layers, train_steps, batch_size, accum_steps]
    # 1e17 4*GPU
    [704, 2048, 11, 9, 1120, 64, 1],
    [640, 1792, 10, 8, 1565, 64, 1],
    [576, 1536, 9, 7, 2281, 64, 1],
    [512, 1472, 8, 6, 3202, 64, 1],
    [448, 1344, 7, 6, 4061, 64, 1],
    # 3e17 4*GPU
    [896, 2240, 14, 11, 1878, 64, 1],
    [768, 2048, 12, 10, 2695, 64, 1],
    [704, 1728, 11, 9, 3763, 64, 1],
    [640, 1728, 10, 7, 5498, 64, 1],
    [576, 1536, 9, 7, 6844, 64, 1],
]

experiment_configs = []
for i, (d_model, d_ff, num_heads, num_layers, train_steps, batch_size, accum_steps) in enumerate(MANUAL_EXPERIMENTS):
    config = {
        'experiment_id': i + 1,
        'model.d_model': d_model,
        'model.d_ff': d_ff,
        'model.num_layers': num_layers,
        'model.num_heads': num_heads,
        'training.train_steps': train_steps,
        # Estimate parameter count: (4 * d_model^2 + 3 * d_model * d_ff) * num_layers
        'model_size_params': (4 * d_model ** 2 + 3 * d_model * d_ff) * num_layers,
        # Estimate token count: train_steps * 262144
        'training_tokens': train_steps * 262144,
        'batch_size': 262144,
        'iterations': train_steps,
        'training.train_batch_size': batch_size,
        'training.gradient_accumulation_steps': accum_steps,
        'training.eval_batch_size': batch_size,
    }
    experiment_configs.append(config)

print(f"Generated {len(experiment_configs)} experiment configs")

# Save experiment configs to CSV
csv_path = os.path.join('pretrain', 'scaling_law_results.csv')
with open(csv_path, 'a', newline='') as f:
    fieldnames = ['experiment_id', 'model.d_model', 'model.d_ff', 'model.num_layers', 
                  'model.num_heads', 'training.train_steps', 'model_size_params', 
                  'training_tokens', 'batch_size', 'iterations', 'train_loss', 'valid_loss']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run experiments
for i, config in enumerate(experiment_configs):
    print(f"\n{'='*60}")
    print(f"Running experiment {i+1}/{len(experiment_configs)}: Experiment ID {config['experiment_id']}")
    print(f"Model params: {config['model_size_params']/1e6:.0f}M")
    print(f"Training tokens: {config['training_tokens']/1e6:.0f}M")
    print(f"Training steps: {config['training.train_steps']}")
    print(f"Model config: d_model={config['model.d_model']}, d_ff={config['model.d_ff']}, "
          f"layers={config['model.num_layers']}, heads={config['model.num_heads']}")

    # Build command
    cmd = [
        'uv', 'run',
        'torchrun', '--standalone', '--nproc_per_node=4', TRAIN_SCRIPT,
        f'--config-name={CONFIG_NAME}',
        f'model.d_model={config["model.d_model"]}',
        f'model.d_ff={config["model.d_ff"]}',
        f'model.num_layers={config["model.num_layers"]}',
        f'model.num_heads={config["model.num_heads"]}',
        f'training.train_steps={config["training.train_steps"]}',
        f'training.train_batch_size={config["training.train_batch_size"]}',
        f'training.gradient_accumulation_steps={config["training.gradient_accumulation_steps"]}',
        f'training.eval_batch_size={config["training.eval_batch_size"]}'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run training script
    try:
        output_lines = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True  # Ensure correct newline handling
        )

        for line in process.stdout:
            # Handle tqdm special characters
            line = line.replace('\r', '\n')  # Replace \r with \n
            print(line, end='', flush=True)  # Force flush
            output_lines.append(line)
        process.wait()
        output = ''.join(output_lines)
        
        # Extract train loss and valid loss
        train_loss = None
        valid_loss = None
        
        train_match = re.findall(r"Training step [0-9]+, Loss: [0-9\.eE+-]+, Smoothed Loss: ([0-9\.eE+-]+)", output)
        valid_match = re.findall(r"Final estimated validation loss: ([0-9\.eE+-]+)", output)
        
        if train_match:
            train_loss = float(train_match[-1])
        if valid_match:
            valid_loss = float(valid_match[-1])
        
        # Add loss to config
        config['train_loss'] = train_loss
        config['valid_loss'] = valid_loss
        
        print(f"Experiment finished: train_loss={train_loss}, valid_loss={valid_loss}")
        
    except Exception as e:
        print(f"Experiment error: {e}")
        config['train_loss'] = None
        config['valid_loss'] = None
    
    # Save result to CSV immediately
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        config.pop('training.train_batch_size', None)
        config.pop('training.eval_batch_size', None)
        config.pop('training.gradient_accumulation_steps', None)
        
        writer.writerow(config)

print(f"\nAll experiments finished! Results saved to: {csv_path}")

# Print experiment summary
print(f"\n{'='*60}")
print("Experiment summary:")
print(f"Total experiments: {len(experiment_configs)}")
print(f"Param range: {min(c['model_size_params'] for c in experiment_configs)/1e6:.0f}M - "
      f"{max(c['model_size_params'] for c in experiment_configs)/1e6:.0f}M")
print(f"Step range: {min(c['training.train_steps'] for c in experiment_configs)} - "
      f"{max(c['training.train_steps'] for c in experiment_configs)}")
