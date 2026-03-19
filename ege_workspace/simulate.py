import torch
import time
import math
import sys

# Parameters
BATCH_SIZE = 128
VOCAB_SIZE = 128256
ALPHA = 16.0
TARGET_TRIGGER = 100000.0
LOG_TARGET_TRIGGER = math.log(TARGET_TRIGGER)
NUM_SIMULATIONS = 10000

# Calibrated Spike Magnitudes
SAFE_SPIKE = 14.2781
ABYSS_SPIKE = 15.0806

# Distributions for W_strain
SAFE_W_MU = 1.0
SAFE_W_SIGMA = 0.05
ABYSS_W_MU = 1.0
ABYSS_W_SIGMA = 0.15

def generate_and_test(num_batches, spike, w_mu, w_sigma, device, dtype):
    triggers = 0
    total_latency = 0.0

    # Process batch by batch to tightly bound memory usage
    for i in range(num_batches):
        if num_batches > 1000 and i % 100 == 0:
            print(f"  ...processed {i}/{num_batches} batches")

        # 1. Generate raw logits (do not pre-allocate huge tensors, do it one batch at a time)
        logits_fp32 = torch.randn((BATCH_SIZE, VOCAB_SIZE), device=device, dtype=torch.float32)
        spike_indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,), device=device)
        logits_fp32[torch.arange(BATCH_SIZE), spike_indices] += spike

        # Cast to target datatype before the actual timing starts
        logits = logits_fp32.to(dtype)

        # Pre-generate W_strain norm to exclude its RNG from latency
        w_strain_buffer = torch.normal(mean=w_mu, std=w_sigma, size=(BATCH_SIZE,), device=device, dtype=torch.float32)
        W_strain = torch.abs(w_strain_buffer).to(dtype)

        # ----------------------------------------------------
        # START TIMING OVERHEAD FOR THE GATE COMPUTATION ONLY
        # ----------------------------------------------------
        start_time = time.perf_counter()

        # 2. Apply softmax and compute entropy
        # Use float32 for stable intermediate values
        logits_fp32_compute = logits.to(torch.float32)
        probs = torch.nn.functional.softmax(logits_fp32_compute, dim=-1)

        # Avoid log(0) NaNs
        safe_probs = torch.clamp(probs, min=1e-10)
        log_probs = torch.log(safe_probs)

        H_x = -torch.sum(probs * log_probs, dim=-1)
        H_x = H_x.to(dtype)

        # 3. Calculate Optimized Metric in Log-Space
        H_x_fp32 = H_x.to(torch.float32)
        W_strain_fp32 = W_strain.to(torch.float32)

        log_S_gate = (ALPHA / H_x_fp32) - torch.log(W_strain_fp32)

        # Cast result back to native hardware dtype
        log_S_gate = log_S_gate.to(dtype)

        # 4. Check Gate Triggers
        batch_triggers = torch.sum(log_S_gate.to(torch.float32) > LOG_TARGET_TRIGGER).item()

        end_time = time.perf_counter()
        # ----------------------------------------------------
        # END TIMING
        # ----------------------------------------------------

        total_latency += (end_time - start_time) * 1000.0
        triggers += batch_triggers

    return triggers, total_latency

def run_simulation(num_sims=NUM_SIMULATIONS):
    torch.set_num_threads(8) # Use all CPU threads to speed up single-process math operations

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    print(f"Running EGE Simulation on device: {device}, dtype: {dtype}, threads: {torch.get_num_threads()}")
    print(f"Total Simulations: {num_sims} batches (Batch Size: {BATCH_SIZE})")

    print(f"Simulating {num_sims//2} 'Safe' distribution batches...")
    safe_triggers, safe_latency = generate_and_test(num_sims // 2, SAFE_SPIKE, SAFE_W_MU, SAFE_W_SIGMA, device, dtype)

    print(f"Simulating {num_sims//2} 'Abyss' distribution batches...")
    abyss_triggers, abyss_latency = generate_and_test(num_sims // 2, ABYSS_SPIKE, ABYSS_W_MU, ABYSS_W_SIGMA, device, dtype)

    # Summary Statistics
    total_latency_ms = safe_latency + abyss_latency
    mean_latency_ms = total_latency_ms / num_sims
    per_token_latency_ms = mean_latency_ms / BATCH_SIZE

    total_abyss_tokens = (num_sims // 2) * BATCH_SIZE
    total_safe_tokens = (num_sims // 2) * BATCH_SIZE

    trigger_rate = (abyss_triggers / total_abyss_tokens) * 100.0 if total_abyss_tokens > 0 else 0
    false_positive_rate = (safe_triggers / total_safe_tokens) * 100.0 if total_safe_tokens > 0 else 0

    print("\n--- EGE Simulation Results ---")
    print(f"Total Simulations: {num_sims} batches (Batch Size: {BATCH_SIZE})")
    print(f"Hardware Emulation Dtype: {dtype}")
    print(f"Optimized Metric: log(S_gate) = {ALPHA} / H(x) - log(||W_strain||)")
    print(f"Trigger Threshold: log({TARGET_TRIGGER}) ≈ {LOG_TARGET_TRIGGER:.4f}")
    print(f"True Positive Trigger Rate (Abyss): {trigger_rate:.4f}% ({abyss_triggers}/{total_abyss_tokens})")
    print(f"False Positive Rate (Safe): {false_positive_rate:.4f}% ({safe_triggers}/{total_safe_tokens})")
    print(f"Mean Batch Gate Computation Latency: {mean_latency_ms:.4f} ms")
    print(f"Mean Per-Token Gate Latency Overhead: {per_token_latency_ms:.6f} ms")

if __name__ == "__main__":
    torch.manual_seed(1337)

    # Allow command line argument to shrink test for CPU, defaulting to 100 batches for CPU testing
    # but still keep the original NUM_SIMULATIONS parameter definition intact.
    num_sims = 100
    if len(sys.argv) > 1:
        num_sims = int(sys.argv[1])

    run_simulation(num_sims)
