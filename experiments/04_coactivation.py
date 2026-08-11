"""
04_coactivation.py — activation-based redundancy: two units that fire on the
same tokens are candidates for pruning even if their weights don't look
similar. Demonstrates why the firing-rate ceiling in CoActivationScanner
matters, using synthetic activations (always available, no --smoke needed).

    python experiments/04_coactivation.py
"""
import torch

from prunelib import CoActivationScanner


def main():
    torch.manual_seed(0)
    num_units, num_tokens = 8, 200

    mask = torch.rand(num_units, num_tokens) > 0.7  # most units fire ~30% of the time

    # Plant one deliberately redundant pair (identical firing pattern).
    mask[1] = mask[0]

    # Plant a near-universal-firing unit -- this is the case the ceiling exists for.
    mask[5] = torch.rand(num_tokens) > 0.02  # fires on ~98% of tokens

    scanner_no_filter = CoActivationScanner(firing_rate_ceiling=1.0)  # effectively no filter
    scanner_filtered = CoActivationScanner(firing_rate_ceiling=0.9)

    sim_no_filter = scanner_no_filter.jaccard(mask)
    sim_filtered = scanner_filtered.jaccard(mask)

    print("firing rates:", [f"{r:.2f}" for r in scanner_filtered.firing_rate(mask).tolist()])
    print(f"\nunit 5 fires on {scanner_filtered.firing_rate(mask)[5]:.0%} of tokens (near-universal)")
    print(f"unit 5's average similarity WITHOUT the ceiling: {sim_no_filter[5].mean():.3f}")
    print(f"unit 5's average similarity WITH the ceiling:    {sim_filtered[5].mean():.3f}  (excluded -> 0)")

    print(f"\nplanted true duplicate (units 0, 1) similarity: {sim_filtered[0, 1]:.3f}  (should be 1.0)")
    print("This is the false-redundancy class the ceiling exists to remove: a")
    print("unit that fires almost everywhere overlaps with almost everything by")
    print("construction, which is not the same thing as being functionally redundant.")


if __name__ == "__main__":
    main()
