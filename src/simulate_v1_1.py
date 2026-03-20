import torch
import time
import math
import sys

# Parameters
BATCH_SIZE = 128
VOCAB_SIZE = 128256
EOS_TOKEN_ID = 128009 # Llama-3 <|eot_id|>
TARGET_TRIGGER = 100000.0
LOG_TARGET_TRIGGER = math.log(TARGET_TRIGGER)
NUM_SIMULATIONS = 100  # Number of batches per state for quick validation

# Calibrated Spike Magnitudes (approximate from earlier tests)
SAFE_SPIKE = 14.28
ABYSS_SPIKE = 15.08
FACTUAL_SPIKE = 15.50  # Even sharper confidence than abyss (e.g. knowing a date exactly)
EOS_SPIKE = 16.00      # Extremely sharp confidence at sequence end

# Distributions for W_strain and Semantic Risk R(x)
SAFE_W_MU, SAFE_W_SIGMA = 1.0, 0.05
ABYSS_W_MU, ABYSS_W_SIGMA = 1.0, 0.15
FACTUAL_W_MU, FACTUAL_W_SIGMA = 1.0, 0.02 # Very low strain during factual recall

def run_simulation_state(state_name, num_batches, spike, w_mu, w_sigma, is_eos, is_high_risk, device, dtype):
    v1_triggers = 0
    v1_1_triggers = 0

    # Pre-allocate memory
    w_strain_buffer = torch.empty((BATCH_SIZE,), device=device, dtype=torch.float32)
    logits_buffer = torch.empty((BATCH_SIZE, VOCAB_SIZE), device=device, dtype=torch.float32)

    # Fixed base alpha for V1.0
    alpha_base = 16.0

    for _ in range(num_batches):
        # 1. Generate raw logits
        torch.randn((BATCH_SIZE, VOCAB_SIZE), out=logits_buffer, device=device, dtype=torch.float32)

        if is_eos:
            spike_indices = torch.full((BATCH_SIZE,), EOS_TOKEN_ID, device=device, dtype=torch.long)
        else:
            spike_indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,), device=device)
            # Ensure it's not the EOS token accidentally
            spike_indices[spike_indices == EOS_TOKEN_ID] = 0

        logits_buffer[torch.arange(BATCH_SIZE), spike_indices] += spike
        logits = logits_buffer.to(dtype)

        # Pre-generate W_strain norm
        torch.normal(mean=w_mu, std=w_sigma, size=(BATCH_SIZE,), out=w_strain_buffer)
        W_strain = torch.abs(w_strain_buffer).to(dtype)

        # 2. Apply softmax and compute entropy
        logits_fp32 = logits.to(torch.float32)
        probs = torch.nn.functional.softmax(logits_fp32, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-10)
        log_probs = torch.log(safe_probs)

        H_x = -torch.sum(probs * log_probs, dim=-1)
        H_x_fp32 = H_x.to(torch.float32)
        W_strain_fp32 = W_strain.to(torch.float32)

        # --- V1.0 Logic (Flawed Baseline) ---
        log_S_gate_v1 = (alpha_base / H_x_fp32) - torch.log(W_strain_fp32)
        batch_triggers_v1 = torch.sum(log_S_gate_v1 > LOG_TARGET_TRIGGER).item()
        v1_triggers += batch_triggers_v1

        # --- V1.1 Logic (Optimized) ---

        # 1. EOS Masking: Extract the probability assigned to the EOS token
        P_eos = probs[:, EOS_TOKEN_ID]
        # Suppression factor: As P(EOS) approaches 1, the scalar approaches 0.
        # Use (1 - P_eos)^gamma to aggressively penalize high EOS confidence.
        gamma = 2.0
        eos_suppression = torch.pow((1.0 - P_eos), gamma)

        # 2. Adaptive Alpha (Factual Recall Problem)
        # Simulate an early-layer semantic risk coefficient R(x) \in [0, 1]
        # High risk (Abyss/Hallucination) -> R(x) ~ 1.0 -> alpha_dynamic ~ 16.0
        # Low risk (Factual/Safe) -> R(x) ~ 0.0 -> alpha_dynamic ~ 8.0
        if is_high_risk:
            # e.g., Abstention/Hallucination territory
            R_x = torch.normal(mean=0.9, std=0.05, size=(BATCH_SIZE,), device=device).clamp(0.0, 1.0)
        else:
            # e.g., General Safe chat or Factual Recall
            R_x = torch.normal(mean=0.1, std=0.05, size=(BATCH_SIZE,), device=device).clamp(0.0, 1.0)

        # alpha_dynamic = 8.0 + 8.0 * R(x)
        alpha_dynamic = 8.0 + (8.0 * R_x)

        # Combine V1.1 Equation
        log_S_gate_v1_1 = (alpha_dynamic / H_x_fp32) - torch.log(W_strain_fp32)
        # Apply the organic suppression
        log_S_gate_v1_1_suppressed = log_S_gate_v1_1 * eos_suppression

        batch_triggers_v1_1 = torch.sum(log_S_gate_v1_1_suppressed > LOG_TARGET_TRIGGER).item()
        v1_1_triggers += batch_triggers_v1_1

    total_tokens = num_batches * BATCH_SIZE
    rate_v1 = (v1_triggers / total_tokens) * 100.0
    rate_v1_1 = (v1_1_triggers / total_tokens) * 100.0

    print(f"\n--- {state_name} ---")
    print(f"  V1.0 Triggers (False Positives if not Abyss): {rate_v1:.2f}% ({v1_triggers}/{total_tokens})")
    print(f"  V1.1 Triggers (False Positives if not Abyss): {rate_v1_1:.2f}% ({v1_1_triggers}/{total_tokens})")

def main():
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    print(f"Running V1.1 EGE Simulation on {device} ({dtype})")

    # 1. Safe State
    # Moderate entropy, low risk, normal strain
    run_simulation_state("Safe Distribution", NUM_SIMULATIONS, SAFE_SPIKE, SAFE_W_MU, SAFE_W_SIGMA, is_eos=False, is_high_risk=False, device=device, dtype=dtype)

    # 2. Factual Recall (The Over-generalization flaw)
    # Extremely low entropy (confident), but low risk, extremely low strain variance
    run_simulation_state("Factual Recall", NUM_SIMULATIONS, FACTUAL_SPIKE, FACTUAL_W_MU, FACTUAL_W_SIGMA, is_eos=False, is_high_risk=False, device=device, dtype=dtype)

    # 3. EOS Token Collapse
    # Extremely low entropy (confident), deterministic spike specifically on token 128009
    run_simulation_state("EOS Token Collapse", NUM_SIMULATIONS, EOS_SPIKE, SAFE_W_MU, SAFE_W_SIGMA, is_eos=True, is_high_risk=False, device=device, dtype=dtype)

    # 4. Abyss / Hallucination
    # Low entropy (confident), high risk, high structural strain variance
    run_simulation_state("Abyss Hallucination", NUM_SIMULATIONS, ABYSS_SPIKE, ABYSS_W_MU, ABYSS_W_SIGMA, is_eos=False, is_high_risk=True, device=device, dtype=dtype)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
