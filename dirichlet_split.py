# dirichlet_split.py
import numpy as np


def dirichlet_label_split(y: np.ndarray, num_clients: int, alpha: float, seed: int = 0):
    """
    Split indices into num_clients using Dirichlet label skew.

    y: array of labels (len = num_samples)
    returns: list[list[int]] client_indices
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    num_classes = int(y.max()) + 1

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # all indicies for class c 
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        # proportions over clients for this class
        # Generates the (p1c, p2c, ..., pnc) for class c for clients
        p = rng.dirichlet([alpha] * num_clients)

        # turn proportions into counts that sum exactly to len(idx_c)
        counts = np.floor(p * len(idx_c)).astype(int)
        # Vector = How many of this class to each client 

        # (both the diff and the if statements) 
        # fix rounding so sum(counts) == len(idx_c)
        diff = len(idx_c) - counts.sum()
        if diff > 0:
            # add 1 to the clients with largest fractional parts
            frac = (p * len(idx_c)) - np.floor(p * len(idx_c))
            for k in np.argsort(-frac)[:diff]:
                counts[k] += 1
        elif diff < 0:
            # remove 1 from clients with smallest fractional parts (that still have >0)
            frac = (p * len(idx_c)) - np.floor(p * len(idx_c))
            for k in np.argsort(frac):
                if diff == 0:
                    break
                if counts[k] > 0:
                    counts[k] -= 1
                    diff += 1

        start = 0
        for k in range(num_clients):
            take = counts[k]
            if take > 0:
                client_indices[k].extend(idx_c[start:start + take].tolist())
            start += take

    # shuffle each client’s indices (optional)
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices
