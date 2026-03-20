import torch
import time
import math
import sys

# Parameters
BATCH_SIZE = 128
VOCAB_SIZE = 128256
SEQ_LEN = 1024 # Context window size representing the self-attention dimension L
EOS_TOKEN_ID = 128009 # Llama-3 <|eot_id|>
TARGET_TRIGGER = 100000.0
LOG_TARGET_TRIGGER = math.log(TARGET_TRIGGER)
NUM_SIMULATIONS = 100  # Number of batches per state for quick validation

# Calibrated Spike Magnitudes (Final Layer Logits)
SAFE_SPIKE = 14.28
ABYSS_SPIKE = 15.08
FACTUAL_SPIKE = 15.50
EOS_SPIKE = 16.00

# Mid-Layer Attention (N/2) "Searching" Spikes
SAFE_ATTN_SPIKE = 4.0      # Normal attention
ABYSS_ATTN_SPIKE = 1.0     # Very diffuse attention (searching/uncertain)
FACTUAL_ATTN_SPIKE = 12.0  # Extremely sharp attention (knows the fact early)
EOS_ATTN_SPIKE = 5.0       # Normal/Sharp attention on context

# Distributions for W_strain
SAFE_W_MU, SAFE_W_SIGMA = 1.0, 0.05
ABYSS_W_MU, ABYSS_W_SIGMA = 1.0, 0.15
FACTUAL_W_MU, FACTUAL_W_SIGMA = 1.0, 0.02

def run_simulation_state(state_name, num_batches, final_spike, attn_spike, w_mu, w_sigma, is_eos, device, dtype):
    v1_triggers = 0
    v1_2_triggers = 0

    alpha = 16.0

    for _ in range(num_batches):
        # 1. Simulate Final Layer Logits (H_final)
        logits_fp32 = torch.randn((BATCH_SIZE, VOCAB_SIZE), device=device, dtype=torch.float32)
        if is_eos:
            spike_indices = torch.full((BATCH_SIZE,), EOS_TOKEN_ID, device=device, dtype=torch.long)
        else:
            spike_indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,), device=device)
            spike_indices[spike_indices == EOS_TOKEN_ID] = 0

        logits_fp32[torch.arange(BATCH_SIZE), spike_indices] += final_spike
        logits = logits_fp32.to(dtype)

        # 2. Simulate Mid-Layer Attention Weights (H_attn)
        attn_fp32 = torch.randn((BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.float32)
        attn_spike_indices = torch.randint(0, SEQ_LEN, (BATCH_SIZE,), device=device)
        attn_fp32[torch.arange(BATCH_SIZE), attn_spike_indices] += attn_spike
        attn_weights = attn_fp32.to(dtype)

        w_strain_buffer = torch.normal(mean=w_mu, std=w_sigma, size=(BATCH_SIZE,), device=device, dtype=torch.float32)
        W_strain = torch.abs(w_strain_buffer).to(dtype)

        # Calculate H_final
        logits_fp32 = logits.to(torch.float32)
        probs = torch.nn.functional.softmax(logits_fp32, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-10)
        log_probs = torch.log(safe_probs)
        H_final = -torch.sum(probs * log_probs, dim=-1)

        # Calculate H_attn (Zero-Overhead Proxy)
        attn_fp32 = attn_weights.to(torch.float32)
        attn_probs = torch.nn.functional.softmax(attn_fp32, dim=-1)
        safe_attn_probs = torch.clamp(attn_probs, min=1e-10)
        log_attn_probs = torch.log(safe_attn_probs)
        H_attn = -torch.sum(attn_probs * log_attn_probs, dim=-1)

        H_final_fp32 = H_final.to(torch.float32)
        H_attn_fp32 = H_attn.to(torch.float32)
        W_strain_fp32 = W_strain.to(torch.float32)

        # --- V1.0 Logic (Flawed Baseline) ---
        log_S_gate_v1 = (alpha / H_final_fp32) - torch.log(W_strain_fp32)
        batch_triggers_v1 = torch.sum(log_S_gate_v1 > LOG_TARGET_TRIGGER).item()
        v1_triggers += batch_triggers_v1

        # --- Stage 6 Logic (Deep-Layer Trajectory) ---

        # Cross-Layer Variance Penalty
        # High ratio (diffuse early, confident late) -> Abyss Hallucination
        # Low ratio (confident early, confident late) -> Factual Recall
        collapse_ratio = H_attn_fp32 / (H_final_fp32 + 1e-5)

        # Calculate delta_H to drop to 0.0 for Factual Recall and scale up linearly for Abyss
        # We clamp at 0.0 so we don't end up with negative numbers
        # When delta_H goes to 0, log_S_gate effectively bypasses the threshold organically
        delta_H = torch.clamp((collapse_ratio - 2.5) * 1.5, min=0.0)

        dynamic_alpha = alpha * delta_H

        # EOS Masking
        P_eos = probs[:, EOS_TOKEN_ID]
        gamma = 2.0
        eos_suppression = torch.pow((1.0 - P_eos), gamma)

        # Combine Stage 6 Equation
        # Adding a small epsilon to prevent completely zeroing out the numerator which might cause log issues down the line
        log_S_gate_v1_2 = ((dynamic_alpha + 1e-5) / H_final_fp32) - torch.log(W_strain_fp32)
        log_S_gate_v1_2_suppressed = log_S_gate_v1_2 * eos_suppression

        batch_triggers_v1_2 = torch.sum(log_S_gate_v1_2_suppressed > LOG_TARGET_TRIGGER).item()
        v1_2_triggers += batch_triggers_v1_2

    total_tokens = num_batches * BATCH_SIZE
    rate_v1 = (v1_triggers / total_tokens) * 100.0
    rate_v1_2 = (v1_2_triggers / total_tokens) * 100.0

    print(f"\n--- {state_name} ---")
    print(f"  V1.0 Triggers (False Positives if not Abyss): {rate_v1:.2f}%")
    print(f"  Stage 6 Triggers (False Positives if not Abyss): {rate_v1_2:.2f}%")

def main():
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    print(f"Running Stage 6 EGE Simulation on {device} ({dtype})")

    run_simulation_state("Safe Distribution", NUM_SIMULATIONS, SAFE_SPIKE, SAFE_ATTN_SPIKE, SAFE_W_MU, SAFE_W_SIGMA, is_eos=False, device=device, dtype=dtype)
    run_simulation_state("Factual Recall", NUM_SIMULATIONS, FACTUAL_SPIKE, FACTUAL_ATTN_SPIKE, FACTUAL_W_MU, FACTUAL_W_SIGMA, is_eos=False, device=device, dtype=dtype)
    run_simulation_state("EOS Token Collapse", NUM_SIMULATIONS, EOS_SPIKE, EOS_ATTN_SPIKE, SAFE_W_MU, SAFE_W_SIGMA, is_eos=True, device=device, dtype=dtype)
    run_simulation_state("Abyss Hallucination", NUM_SIMULATIONS, ABYSS_SPIKE, ABYSS_ATTN_SPIKE, ABYSS_W_MU, ABYSS_W_SIGMA, is_eos=False, device=device, dtype=dtype)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
