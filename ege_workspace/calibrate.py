import torch

def generate_logits(batch_size, vocab_size, spike_magnitude, device, dtype):
    logits = torch.randn((batch_size, vocab_size), device=device, dtype=dtype)
    spike_indices = torch.randint(0, vocab_size, (batch_size,), device=device)
    logits[torch.arange(batch_size), spike_indices] += spike_magnitude
    return logits

def calculate_entropy(logits, temperature=1.0):
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    # Using more stable log calculation
    # Only calculate log where prob > 0
    safe_probs = torch.clamp(probs, min=1e-10)
    log_probs = torch.log(safe_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def binary_search_spike(target_entropy, batch_size=128, vocab_size=128256, device='cpu', dtype=torch.float32):
    print(f"Finding spike for target H ≈ {target_entropy}")
    low = 0.0
    high = 20.0

    best_spike = 0.0
    best_diff = float('inf')

    for _ in range(50):
        mid = (low + high) / 2
        logits = generate_logits(batch_size, vocab_size, mid, device, dtype)
        current_entropy = calculate_entropy(logits)

        diff = abs(current_entropy - target_entropy)
        if diff < best_diff:
            best_diff = diff
            best_spike = mid

        if current_entropy < target_entropy:
            # lower entropy means sharper distribution, so lower the spike
            high = mid
        else:
            low = mid

    # Final confirmation
    torch.manual_seed(42)
    logits = generate_logits(batch_size, vocab_size, best_spike, device, dtype)
    print(f"Target: {target_entropy}, Found Entropy: {calculate_entropy(logits):.4f} at Spike: {best_spike:.4f}")
    return best_spike

if __name__ == "__main__":
    device = 'cpu'
    dtype = torch.float32  # Using float32 for calibration to avoid bfloat16 precision issues

    safe_target = 2.13
    abyss_target = 1.17

    safe_spike = binary_search_spike(safe_target, device=device, dtype=dtype)
    abyss_spike = binary_search_spike(abyss_target, device=device, dtype=dtype)

    print(f"\nResults:")
    print(f"Safe state (H={safe_target}) spike magnitude: {safe_spike:.4f}")
    print(f"Abyss state (H={abyss_target}) spike magnitude: {abyss_spike:.4f}")
