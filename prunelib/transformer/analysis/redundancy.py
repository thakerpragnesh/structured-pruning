"""Pure analysis over a recorded activation-firing tensor.

Finds neuron pairs whose firing pattern overlaps almost perfectly (true
Jaccard/IoU: intersection over union) -- functionally redundant
regardless of their raw weight saliency. Not from the thesis; a new
criterion explored alongside the weight-based ones.
"""
import torch


class JaccardTwinFinder:
    """Pairwise Jaccard overlap between neurons' firing patterns.

    Dead neurons are identified directly from their firing counts rather
    than from the Jaccard matrix. They cannot be found in that matrix:
    two neurons that never fire have `intersection = union = 0`, and the
    divide-by-zero guard maps that to a similarity of 0, which no
    sensible threshold accepts. Reporting them separately also says the
    more useful thing -- a dead neuron is unconditionally prunable, not
    merely redundant with some particular partner.
    """

    def find(self, fires: torch.Tensor, threshold: float = 0.95, max_pairs: int = None):
        """Return `(twins, dead)`.

        `twins` is a list of `(i, j, overlap)` with `i < j`, sorted by
        descending overlap so a caller taking the top-N takes the most
        redundant ones. `dead` is a sorted list of neurons that never
        fired.

        `fires` is `(tokens, neurons)`, bool or float, as produced by
        `activation_recording.ActivationRecorder`.
        """
        if fires.dim() != 2:
            raise ValueError(f"`fires` must be (tokens, neurons), got shape {tuple(fires.shape)}")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        # float32 counts are exact for integers up to 2**24, far above any
        # realistic token count, so the matmul stays cheap without risking
        # an off-by-one in an intersection count.
        fires = fires.to(torch.float32)
        counts = fires.sum(dim=0)
        alive = counts > 0
        dead = (~alive).nonzero(as_tuple=True)[0].tolist()

        intersection = fires.t() @ fires
        union = counts.unsqueeze(0) + counts.unsqueeze(1) - intersection
        jaccard = intersection / union.clamp(min=1.0)

        # Upper triangle only (the matrix is symmetric, and the diagonal is
        # trivially 1.0), and both neurons must actually fire.
        candidates = torch.triu(jaccard >= threshold, diagonal=1)
        candidates &= alive.unsqueeze(0) & alive.unsqueeze(1)

        rows, cols = candidates.nonzero(as_tuple=True)
        overlaps = jaccard[rows, cols]
        order = torch.argsort(overlaps, descending=True)
        if max_pairs is not None:
            order = order[:max_pairs]

        # One host transfer for the whole result, not one per pair.
        twins = list(zip(rows[order].tolist(), cols[order].tolist(), overlaps[order].tolist()))
        return twins, dead
