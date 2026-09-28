#!/usr/bin/env python
"""Behavior cloning from expert demonstrations.

Loads (obs, action, mask) tuples from an .npz file and trains a policy
via supervised cross-entropy loss on valid actions only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior cloning from expert data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--output", type=str, required=True, help="Output model path (.zip)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--net-arch", type=str, default="residual",
                        help="Network architecture: 'residual' or comma-separated sizes")
    parser.add_argument("--activation", type=str, default="elu", choices=["relu", "tanh", "elu"])
    return parser.parse_args()


def build_net(sizes, activation_fn, obs_dim, action_dim):
    """Build a simple MLP policy network (same structure as SB3's MlpPolicy)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation_fn)
    # Add residual connections for matching sizes
    net = nn.Sequential(*layers)
    # Action head
    action_head = nn.Linear(sizes[-1], action_dim)
    return net, action_head


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(__file__).resolve().parent.parent

    # Load dataset
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = root / data_path
    print(f"Loading dataset from {data_path}...", flush=True)
    data = np.load(str(data_path))
    obs = data["obs"]          # (N, obs_dim)
    actions = data["actions"]  # (N,)
    masks = data["masks"]      # (N, action_dim)

    N, obs_dim = obs.shape
    action_dim = masks.shape[1]
    print(f"  {N} timesteps, obs_dim={obs_dim}, action_dim={action_dim}", flush=True)

    # Build network (matching the residual [256,256,256,256] architecture)
    activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    act_fn = activation_map[args.activation]()

    if args.net_arch == "residual":
        sizes = [256, 256, 256, 256]
    else:
        sizes = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    sizes = [obs_dim] + sizes

    # Build feature extractor + action head
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act_fn)
        # Residual: if input and output sizes match, add residual connection
    feature_net = nn.Sequential(*layers)
    action_head = nn.Linear(sizes[-1], action_dim)

    optimizer = torch.optim.Adam(
        list(feature_net.parameters()) + list(action_head.parameters()),
        lr=args.lr,
    )

    # Convert to tensors
    obs_t = torch.from_numpy(obs).float()
    actions_t = torch.from_numpy(actions).long()
    masks_t = torch.from_numpy(masks).float()  # 1 = valid, 0 = invalid

    dataset_size = N
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Shuffle
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, dataset_size, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch_obs = obs_t[idx]
            batch_actions = actions_t[idx]
            batch_masks = masks_t[idx]

            # Forward
            features = feature_net(batch_obs)
            logits = action_head(features)

            # Mask invalid actions: set logits of invalid actions to -inf
            # masks are 1 for valid, 0 for invalid
            masked_logits = logits.clone()
            masked_logits[~batch_masks.bool()] = -1e9

            # Cross-entropy loss over valid actions only
            loss = F.cross_entropy(masked_logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(feature_net.parameters()) + list(action_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Compute accuracy on a sample
        with torch.no_grad():
            sample_idx = torch.randperm(dataset_size)[:min(5000, dataset_size)]
            sample_obs = obs_t[sample_idx]
            sample_actions = actions_t[sample_idx]
            sample_masks = masks_t[sample_idx]
            sample_features = feature_net(sample_obs)
            sample_logits = action_head(sample_features)
            sample_logits[~sample_masks.bool()] = -1e9
            preds = sample_logits.argmax(dim=1)
            acc = (preds == sample_actions).float().mean().item()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | acc={acc:.3f}", flush=True)

    # Save the trained policy as PyTorch state dicts
    output_dir = root / "models"
    torch.save(feature_net.state_dict(), output_dir / "bc_feature_net.pt")
    torch.save(action_head.state_dict(), output_dir / "bc_action_head.pt")
    print(f"BC weights saved to {output_dir / 'bc_feature_net.pt'}", flush=True)

    # Also create and save a full SB3 model with loaded weights
    print(f"\nCreating SB3 MaskablePPO with BC weights...", flush=True)
    from sb3_contrib import MaskablePPO
    from showdownrl.simple_env import SimplePokemonMoveEnv

    temp_env = SimplePokemonMoveEnv(
        seed=args.seed,
        mechanics="advanced",
        observation_mode="simple",
    )
    model = MaskablePPO(
        "MlpPolicy", temp_env,
        policy_kwargs={
            "net_arch": [256, 256, 256, 256],
            "activation_fn": activation_map[args.activation],
        },
        verbose=0, seed=args.seed,
    )

    # Transfer weights by matching state_dict keys
    with torch.no_grad():
        # Our feature_net is 4 Linear layers [86->256, 256->256, 256->256, 256->256]
        # SB3 policy_net and value_net have the same structure
        our_fw = list(feature_net.parameters())  # 8 tensors: w,b,w,b,w,b,w,b
        sb3_sd = model.policy.state_dict()

        matched = 0
        # Copy to policy_net layers 0,2,4,6 (skipping activations at odd indices)
        for layer_i in range(4):
            for param_name in ['weight', 'bias']:
                our_idx = layer_i * 2  # 0: w, 1: b, 2: w, 3: b, ...
                sb3_key = f"mlp_extractor.policy_net.{layer_i * 2}.{param_name}"
                if our_fw[our_idx].shape == sb3_sd[sb3_key].shape:
                    sb3_sd[sb3_key].copy_(our_fw[our_idx])
                    matched += 1

        # Copy same weights to value_net
        for layer_i in range(4):
            for param_name in ['weight', 'bias']:
                our_idx = layer_i * 2
                sb3_key = f"mlp_extractor.value_net.{layer_i * 2}.{param_name}"
                if our_fw[our_idx].shape == sb3_sd[sb3_key].shape:
                    sb3_sd[sb3_key].copy_(our_fw[our_idx])
                    matched += 1

        # Copy action head
        sb3_sd["action_net.weight"].copy_(action_head.weight.data)
        sb3_sd["action_net.bias"].copy_(action_head.bias.data)
        matched += 2

        # Initialize value_net from action_net (scaled down)
        sb3_sd["value_net.weight"].copy_(action_head.weight.data[:1] * 0.1)
        sb3_sd["value_net.bias"].copy_(action_head.bias.data[:1] * 0.1)
        matched += 2

        print(f"  Transferred {matched} parameter tensors", flush=True)

        # Load modified state dict
        model.policy.load_state_dict(sb3_sd)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    temp_env.close()
    print(f"BC model saved to {output_path}", flush=True)
    print(f"  Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
