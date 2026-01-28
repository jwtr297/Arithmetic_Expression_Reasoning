from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "ModularVocabulary",
    "build_modular_vocabulary",
    "generate_noisy_twosteps_dataset_modp",
]


@dataclass(frozen=True)
class ModularVocabulary:
    """Container describing the token ids used for modular arithmetic data."""

    modulus: int
    tokens: Dict[str, int]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def __getitem__(self, item: str) -> int:
        return self.tokens[item]


def _validate_prime(p: int) -> None:
    if not isinstance(p, int) or p < 2:
        raise ValueError("The modulus p must be an integer >= 2.")
    if p in (2, 3):
        return
    if p % 2 == 0:
        raise ValueError("The modulus p must be prime.")
    limit = int(math.isqrt(p))
    for factor in range(3, limit + 1, 2):
        if p % factor == 0:
            raise ValueError("The modulus p must be prime.")


def build_modular_vocabulary(p: int) -> ModularVocabulary:
    """Create a vocabulary for arithmetic modulo ``p``.

    Parameters
    ----------
    p:
        A prime modulus.  The vocabulary contains integer tokens ``0..p-1``
        plus ``PLUS``, ``MINUS``, ``MUL``, ``LP``, ``RP``, ``DEDUCT`` (``-->``),
        ``END`` (``#``), and ``PAD``.
    """

    #_validate_prime(p)

    tokens: Dict[str, int] = {str(i): i for i in range(p)}
    next_id = p

    def _assign(name: str) -> int:
        nonlocal next_id
        tokens[name] = next_id
        next_id += 1
        return tokens[name]

    _assign("PLUS")
    _assign("MINUS")
    _assign("LP")
    _assign("RP")
    _assign("MUL")
    _assign("DEDUCT")
    _assign("END")
    _assign("PAD")

    return ModularVocabulary(modulus=p, tokens=tokens)






def _sample_different(rng: random.Random, low: int, high: int, *, exclude: int) -> int:
    """Sample a value in ``[low, high]`` that differs from ``exclude``.

    If ``exclude`` is the only value in the interval, the original value is
    returned to avoid an infinite loop.  This can happen for ``p = 2`` when the
    valid operand set ``{0}`` contains a single element.
    """

    if low == high == exclude:
        return exclude

    while True:
        value = rng.randint(low, high)
        if value != exclude:
            return value


def generate_noisy_twosteps_dataset_modp(
    N: int,
    p: int,                     # modulus (should be prime; code works for any p>=2)
    p1: float,                  # probability to corrupt exactly one of {a,b,c}
    p2: float,                  # probability to corrupt e
    seed: int = None,
    device: str = "cpu",
    pad_to_len: int = 48,       # total sequence length with left padding
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    assert p >= 2, "p must be >= 2" 
    dev = torch.device(device)

    # ------------------- token ID scheme -------------------
    
    PLUS_ID   = p
    MINUS_ID  = p + 1
    LP_ID     = p + 2
    RP_ID     = p + 3
    MUL_ID    = p + 4
    SEP_ID    = p + 5
    EOS_ID    = p + 6
    PAD_ID    = p + 7
    VOCAB_SIZE = p + 8

    # ------------------- helpers -------------------
    def rand_pm_token() -> int:
        """Randomly return '+' or '-' token ID."""
        return PLUS_ID if random.random() < 0.5 else MINUS_ID

    def sample_digit_excluding(exclude: int) -> int:
        """Sample a digit in {0,...,p-1} different from `exclude`."""
        # Draw uniformly from p-1 values; avoid rejection loops for speed.
        r = random.randint(0, p - 2)
        return r if r < exclude else r + 1

    def modp(x: int) -> int:
        """Modulo p in [0..p-1]."""
        return x % p

    # ------------------- seeding -------------------
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # ------------------- layout lengths -------------------
    BASE_LEN = 16  # fixed by template (see docstring)
    T = pad_to_len
    assert T >= BASE_LEN, "pad_to_len must be at least BASE_LEN=16"
    

    # Prepare output tensor with PAD
    seqs = torch.full((N, T), PAD_ID, dtype=torch.long, device=dev)
    
    vocab = build_modular_vocabulary(p)

    for i in range(N):
        # Sample clean operands (uniform over 0..p-1, including 0 and 1)
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)
        c = random.randint(0, p - 1)
        d = random.randint(0, p - 1)

        # Sample signs
        op1 = rand_pm_token()
        op2 = rand_pm_token()

        # Compute e, r on CLEAN values (mod p)
        s1 = (a + b) if op1 == PLUS_ID else (a - b)
        e  = modp(s1 * c)
        f  = d
        s2 = (e + f) if op2 == PLUS_ID else (e - f)
        r  = modp(s2)

        # Build clean sequence (16 tokens)
        tokens = [
            LP_ID, a, op1, b, RP_ID, MUL_ID, c, op2, d,
            SEP_ID, e, op2, f, SEP_ID, r, EOS_ID
        ]

        # Noise on E1: corrupt exactly one among {a,b,c} with prob p1
        if random.random() < p1:
            idx = random.choice([1, 3, 6])  # positions of a, b, c in template
            old = tokens[idx]
            tokens[idx] = sample_digit_excluding(old)

        # Noise on E2: corrupt e with prob p2
        if random.random() < p2:
            old = tokens[10]  # e
            tokens[10] = sample_digit_excluding(old)

        # Left-pad to length T and write
        seq = torch.tensor(tokens, dtype=torch.long, device=dev)
        seqs[i, -BASE_LEN:] = seq

    # Index of the token 'e' (position 10 of the 0-based template) in the padded sequence
    solution_start = (T - BASE_LEN) + 10

    info = {
        "solution_start_ind": [solution_start] * N,
        "vocab_size": VOCAB_SIZE,
        "token_ids": {
            "PLUS_ID": PLUS_ID, "MINUS_ID": MINUS_ID,
            "LP_ID": LP_ID, "RP_ID": RP_ID,
            "MUL_ID": MUL_ID, "SEP_ID": SEP_ID,
            "EOS_ID": EOS_ID, "PAD_ID": PAD_ID,
        },
        "p": p,
        "noise": {"p1": p1, "p2": p2},
    }
    return seqs, vocab, info




def generate_noisy_twosteps_dataset_modp_1(
    N: int,
    p: int,
    p1: float,
    p2: float,
    *,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    e2_noise_indices: Optional[Sequence[int]] = None,  # kept for API compatibility; ignored (we only corrupt product)
    extended_expression: bool = False,
) -> Tuple[torch.Tensor, "ModularVocabulary", Dict[str, object]]:
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be probabilities in [0, 1].")

    vocab = build_modular_vocabulary(p)

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)

    device = torch.device(device)

    PAD_ID    = vocab["PAD"]
    PLUS_ID   = vocab["PLUS"]
    MINUS_ID  = vocab["MINUS"]
    MUL_ID    = vocab["MUL"]
    LP_ID     = vocab["LP"]
    RP_ID     = vocab["RP"]
    DEDUCT_ID = vocab["DEDUCT"]
    END_ID    = vocab["END"]

    # Sequence lengths and solution start indices depend on layout
    if extended_expression:
        base_len = 16
        solution_start_index = 10  # index right after the first DEDUCT
        product_idx_in_e2 = 10     # product position inside E2 for extended layout
        e1_noise_candidates = (1, 3)  # only a (idx=1) or b (idx=3)
    else:
        base_len = 14
        solution_start_index = 8
        product_idx_in_e2 = 8      # product position inside E2 for short layout
        e1_noise_candidates = (1, 3)  # only a (idx=1) or b (idx=3)

    seqs = torch.full((N, base_len), PAD_ID, dtype=torch.long, device=device)

    for i in range(N):
        if extended_expression:
            # Sample operands
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            d = rng.randint(0, p - 1)
            first_op  = PLUS_ID if rng.random() < 0.5 else MINUS_ID
            second_op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # Compute exact (noise-free) intermediates / result
            first_partial = (a + b) % p if first_op == PLUS_ID else (a - b) % p
            product = (first_partial * c) % p
            result  = (product + d) % p if second_op == PLUS_ID else (product - d) % p

            # Build token sequence
            tokens: List[int] = [
                LP_ID, a, first_op, b, RP_ID, MUL_ID, c, second_op, d,
                DEDUCT_ID,
                product,
                second_op, d,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either a or b (not c/d)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {1, 3}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        else:
            # Short layout: (a * b) +/- c
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            product = (a * b) % p
            result  = (product + c) % p if op == PLUS_ID else (product - c) % p

            tokens = [
                LP_ID, a, MUL_ID, b, RP_ID, op, c,
                DEDUCT_ID,
                product,
                op, c,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either a or b (not c)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {1, 3}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        # E2 noise: ONLY corrupt the product token
        if rng.random() < p2:
            tokens[product_idx_in_e2] = _sample_different(
                rng, 0, p - 1, exclude=tokens[product_idx_in_e2]
            )

        # Write sequence
        seqs[i] = torch.tensor(tokens, dtype=torch.long, device=device)

    info = {
        "solution_start_ind": [solution_start_index] * N,
        "vocab_size": vocab.size,
        "modulus": p,
        "base_length": base_len,
    }

    return seqs, vocab, info










def generate_noisy_twosteps_varied_dataset_modp_og(
    N: int,
    p: int,
    p1: float,
    p2: float,
    *,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    e2_noise_indices: Optional[Sequence[int]] = None,  # kept for API compatibility; ignored
    extended_expression: bool = False,
) -> Tuple[torch.Tensor, "ModularVocabulary", Dict[str, object]]:
   
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be probabilities in [0, 1].")

    vocab = build_modular_vocabulary(p)

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)

    device = torch.device(device)

    PAD_ID    = vocab["PAD"]
    PLUS_ID   = vocab["PLUS"]
    MINUS_ID  = vocab["MINUS"]
    MUL_ID    = vocab["MUL"]
    LP_ID     = vocab["LP"]
    RP_ID     = vocab["RP"]
    DEDUCT_ID = vocab["DEDUCT"]
    END_ID    = vocab["END"]

    # Sequence lengths and solution start indices depend on layout
    if extended_expression:
        base_len = 16
        solution_start_index = 10  # index right after the first DEDUCT
        product_idx_in_e2 = 10     # product position inside E2 for extended layout
        # E1 noise: only a (idx=1) or b (idx=3)
        e1_noise_candidates = (1, 3)
    else:
        base_len = 14
        solution_start_index = 8   # first token after the first DEDUCT
        product_idx_in_e2 = 8      # here this is the intermediate d in a * d
        # New short layout E1 noise: only b or c can be corrupted
        # In the token layout below, b is at index 3, c at index 5.
        e1_noise_candidates = (3, 5)

    seqs = torch.full((N, base_len), PAD_ID, dtype=torch.long, device=device)

    for i in range(N):
        if extended_expression:
            # Extended layout: (a +/- b) * c +/- d  (unchanged)
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            d = rng.randint(0, p - 1)
            first_op  = PLUS_ID if rng.random() < 0.5 else MINUS_ID
            second_op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # Compute exact (noise-free) intermediates / result
            first_partial = (a + b) % p if first_op == PLUS_ID else (a - b) % p
            product = (first_partial * c) % p
            result  = (product + d) % p if second_op == PLUS_ID else (product - d) % p

            # Build token sequence
            tokens: List[int] = [
                LP_ID, a, first_op, b, RP_ID, MUL_ID, c, second_op, d,
                DEDUCT_ID,
                product,
                second_op, d,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either a or b (not c/d)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {1, 3}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        else:
            # New short layout: a * (b +/- c) --> a * d --> r
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # d = (b +/- c) mod p, r = (a * d) mod p
            d = (b + c) % p if op == PLUS_ID else (b - c) % p
            result = (a * d) % p

            # Token layout (indices):
            #  0: a
            #  1: MUL
            #  2: LP
            #  3: b
            #  4: op
            #  5: c
            #  6: RP
            #  7: DEDUCT
            #  8: d          <-- product_idx_in_e2 points here
            #  9: MUL
            # 10: a
            # 11: DEDUCT
            # 12: result
            # 13: END
            tokens = [
                a, MUL_ID, LP_ID, b, op, c, RP_ID,
                DEDUCT_ID,
                d,
                MUL_ID, a,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either b or c (not a)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {3, 5}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        # E2 noise: ONLY corrupt the product / intermediate token
        if rng.random() < p2:
            tokens[product_idx_in_e2] = _sample_different(
                rng, 0, p - 1, exclude=tokens[product_idx_in_e2]
            )

        # Write sequence
        seqs[i] = torch.tensor(tokens, dtype=torch.long, device=device)

    info = {
        "solution_start_ind": [solution_start_index] * N,
        "vocab_size": vocab.size,
        "modulus": p,
        "base_length": base_len,
    }

    return seqs, vocab, info








def generate_noisy_twosteps_varied_dataset_modp(
    N: int,
    p: int,
    p1: float,
    p2: float,
    *,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    e2_noise_indices: Optional[Sequence[int]] = None,  # kept for API compatibility; ignored
    extended_expression: bool = False,
) -> Tuple[torch.Tensor, "ModularVocabulary", Dict[str, object]]:
    
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be probabilities in [0, 1].")

    vocab = build_modular_vocabulary(p)

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)

    device = torch.device(device)

    PAD_ID    = vocab["PAD"]
    PLUS_ID   = vocab["PLUS"]
    MINUS_ID  = vocab["MINUS"]
    MUL_ID    = vocab["MUL"]
    LP_ID     = vocab["LP"]
    RP_ID     = vocab["RP"]
    DEDUCT_ID = vocab["DEDUCT"]
    END_ID    = vocab["END"]

    # Sequence lengths and solution start indices depend on layout
    if extended_expression:
        base_len = 16
        solution_start_index = 10  # index right after the first DEDUCT
        product_idx_in_e2 = 10     # product position inside E2 for extended layout
        # E1 noise: only a (idx=1) or b (idx=3)
        e1_noise_candidates = (1, 3)
    else:
        base_len = 14
        solution_start_index = 8   # first token after the first DEDUCT
        # After swapping positions in E2, d is now at index 10
        product_idx_in_e2 = 10     # this is the intermediate d in a * d (E2)
        # New short layout E1 noise: only b or c can be corrupted
        # In the token layout below, b is at index 3, c at index 5.
        e1_noise_candidates = (3, 5)

    seqs = torch.full((N, base_len), PAD_ID, dtype=torch.long, device=device)

    for i in range(N):
        if extended_expression:
            # Extended layout: (a +/- b) * c +/- d  (unchanged)
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            d = rng.randint(0, p - 1)
            first_op  = PLUS_ID if rng.random() < 0.5 else MINUS_ID
            second_op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # Compute exact (noise-free) intermediates / result
            first_partial = (a + b) % p if first_op == PLUS_ID else (a - b) % p
            product = (first_partial * c) % p
            result  = (product + d) % p if second_op == PLUS_ID else (product - d) % p

            # Build token sequence
            tokens: List[int] = [
                LP_ID, a, first_op, b, RP_ID, MUL_ID, c, second_op, d,
                DEDUCT_ID,
                product,
                second_op, d,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either a or b (not c/d)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {1, 3}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        else:
            # New short layout: a * (b +/- c) --> a * d --> r
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # d = (b +/- c) mod p, r = (a * d) mod p
            d = (b + c) % p if op == PLUS_ID else (b - c) % p
            result = (a * d) % p

            # Token layout (indices) AFTER swapping a and d in E2:
            #  0: a
            #  1: MUL
            #  2: LP
            #  3: b
            #  4: op
            #  5: c
            #  6: RP
            #  7: DEDUCT
            #  8: a
            #  9: MUL
            # 10: d          <-- product_idx_in_e2 points here
            # 11: DEDUCT
            # 12: result
            # 13: END
            
            tokens = [
                a, MUL_ID, LP_ID, b, op, c, RP_ID,
                DEDUCT_ID,
                a,
                MUL_ID, d,
                DEDUCT_ID,
                result,
                END_ID,
            ]

            # E1 noise: corrupt either b or c (not a)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {3, 5}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        # E2 noise: ONLY corrupt the product / intermediate token (d)
        if rng.random() < p2:
            tokens[product_idx_in_e2] = _sample_different(
                rng, 0, p - 1, exclude=tokens[product_idx_in_e2]
            )

        # Write sequence
        seqs[i] = torch.tensor(tokens, dtype=torch.long, device=device)

    info = {
        "solution_start_ind": [solution_start_index] * N,
        "vocab_size": vocab.size,
        "modulus": p,
        "base_length": base_len,
    }

    return seqs, vocab, info













def compute_short_E3_distributions(
    tokens: torch.Tensor,
    p: int,
    p1: float,
    p2: float,
    plus_id: int | None = None,   
    minus_id: int | None = None,  
):
    """
    Analytic population distributions for SHORT layout over a batch.

    Layout (short):
      [LP, a, MUL, b, RP, op, c, DEDUCT, product, op, c, DEDUCT, result, END]

    Return:
      P_E3_given_E1, P_E3_given_E2, P_E3_given_E1E2  (each [N, p], rows sum to 1)
    """
    # ---------- checks & setup ----------
    assert tokens.ndim == 2, "tokens must be [N, L]"
    p = int(p); assert p >= 2
    N, L = tokens.shape
    device = tokens.device
    dtype = torch.float64
    eps = 1e-12

    PLUS_ID  = p if plus_id  is None else int(plus_id)
    MINUS_ID = p+1 if minus_id is None else int(minus_id)

    # ---------- extract fields ----------
    a1    = tokens[:, 1].long()
    b1    = tokens[:, 3].long()
    op    = tokens[:, 5].long()
    c     = tokens[:, 6].long()
    m_obs = tokens[:, 8].long()

    # op_sign ∈ {+1,-1}
    op_sign = torch.where(op == PLUS_ID, torch.ones_like(c), -torch.ones_like(c))
    shift = (op_sign * c) % p  # [N] long

    # ---------- P(m | E1) ----------
    pm_E1 = torch.zeros((N, p), dtype=dtype, device=device)
    m0 = (a1 * b1) % p

    both_nz   = (a1 != 0) & (b1 != 0)
    exactly_1 = (a1 == 0) ^ (b1 == 0)
    both_0    = (a1 == 0) & (b1 == 0)

    # A) both non-zero: mass 1-p1 at m0, uniform p1 over others
    if both_nz.any():
        rows = torch.nonzero(both_nz, as_tuple=True)[0]
        pm_E1[rows, :] = (p1 / (p - 1))
        pm_E1[rows, m0[rows]] = 1.0 - p1

    # B) exactly one zero: mass 1-p1/2 at 0, uniform (p1/2) over nonzeros
    if exactly_1.any():
        rows = torch.nonzero(exactly_1, as_tuple=True)[0]
        pm_E1[rows, :] = (p1 / 2.0) / (p - 1)
        pm_E1[rows, 0] = 1.0 - p1 / 2.0

    # C) both zero: all mass at 0
    if both_0.any():
        rows = torch.nonzero(both_0, as_tuple=True)[0]
        pm_E1[rows, :] = 0.0
        pm_E1[rows, 0] = 1.0

    # safety renorm
    s = pm_E1.sum(dim=1, keepdim=True)
    pm_E1 = pm_E1 / (s + eps)

    # ---------- prior P(m) ----------
    prior_m = torch.full((p,), (p - 1) / (p * p), dtype=dtype, device=device)
    prior_m[0] = (2 * p - 1) / (p * p)
    prior_row = prior_m.view(1, -1).expand(N, -1)

    # ---------- likelihood L = P(E2=m_obs | m) ----------
    Lk = torch.full((N, p), p2 / (p - 1), dtype=dtype, device=device)
    Lk.scatter_(1, m_obs.view(-1, 1), (1.0 - p2) * torch.ones((N, 1), dtype=dtype, device=device))

    # ---------- P(m | E2) ----------
    un_E2 = prior_row * Lk
    pm_E2 = un_E2 / (un_E2.sum(dim=1, keepdim=True) + eps)

    # ---------- P(m | E1, E2) (E2 ⟂ E1 | m) ----------
    un_E1E2 = pm_E1 * Lk
    pm_E1E2 = un_E1E2 / (un_E1E2.sum(dim=1, keepdim=True) + eps)

    # ---------- map m -> E3 by per-row shift ----------
    # E3[z] = P(m = (z - shift) mod p)
    cols = torch.arange(p, device=device).view(1, -1).expand(N, -1)        # [N, p]
    idx  = (cols - shift.view(-1, 1)) % p                                   # [N, p], long

    P_E3_given_E1   = pm_E1.gather(1, idx)
    P_E3_given_E2   = pm_E2.gather(1, idx)
    P_E3_given_E1E2 = pm_E1E2.gather(1, idx)

    # final safety renorm 
    for T in (P_E3_given_E1, P_E3_given_E2, P_E3_given_E1E2):
        T.clamp_(min=0.0)
        T /= (T.sum(dim=1, keepdim=True) + eps)

    return P_E3_given_E1, P_E3_given_E2, P_E3_given_E1E2






@torch.no_grad()
def compute_e1_e2_distributions(
    seqs: torch.LongTensor,
    p: int,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Token scheme:
      0..p-1: digits
      p    : PLUS
      p+1  : MINUS
      p+2  : LP
      p+3  : RP
      p+4  : MUL
      p+5  : DEDUCT
      p+6  : END
      p+7  : PAD

    Output:
      e1_dist: one-hot over the FULL E1 expression value BEFORE the first DEDUCT.
               - short:    (a * b) +/- c
               - extended: (a +/- b) * c +/- d
      e2_dist: one-hot over the result recomputed from tokens AFTER first DEDUCT:
               [product, op, operand]  -> final result
    """
    assert seqs.dim() == 2 and seqs.dtype == torch.long, "seqs must be LongTensor [N, L]"
    N, L = seqs.shape
    device = seqs.device

    PLUS   = p
    MINUS  = p + 1
    LP     = p + 2
    MUL    = p + 4
    DEDUCT = p + 5

    e1_dist = torch.zeros((N, p), dtype=torch.float, device=device)
    e2_dist = torch.zeros((N, p), dtype=torch.float, device=device)

    for i in range(N):
        row = seqs[i]

        # ---- locate first DEDUCT (separator between E1-part and E2-part) ----
        d_pos = (row == DEDUCT).nonzero(as_tuple=False)
        if d_pos.numel() == 0:
            continue
        d0 = int(d_pos[0].item())

        # ==================== E2 ====================
        # After first DEDUCT: [product, op2, operand, DEDUCT, ...]
        if d0 + 3 < L:
            e2_product = int(row[d0 + 1].item())
            e2_op      = int(row[d0 + 2].item())
            e2_operand = int(row[d0 + 3].item())
            if (0 <= e2_product < p) and (0 <= e2_operand < p) and (e2_op in (PLUS, MINUS)):
                e2_res = (e2_product + e2_operand) % p if e2_op == PLUS else (e2_product - e2_operand) % p
                e2_dist[i, e2_res] = 1.0

        # ==================== E1 (FULL expr before first DEDUCT) ====================
        # Find first '(' before DEDUCT
        lp_pos = (row[:d0] == LP).nonzero(as_tuple=False)
        if lp_pos.numel() == 0:
            continue
        lp = int(lp_pos[0].item())
        if lp + 3 >= d0:
            continue

        a = int(row[lp + 1].item())
        op_or_mul = int(row[lp + 2].item())
        b = int(row[lp + 3].item())

        # Validate a, b as digits
        if not (0 <= a < p and 0 <= b < p):
            continue

        if op_or_mul == MUL:
            # ---------- short layout: (a * b) +/- c ----------
            # Expected tokens before DEDUCT:
            # [LP, a, MUL, b, RP, op1, c, DEDUCT, ...]
            # So op1 at lp+5, c at lp+6
            if lp + 6 >= d0:
                continue
            op1 = int(row[lp + 5].item())
            c   = int(row[lp + 6].item())

            if (op1 in (PLUS, MINUS)) and (0 <= c < p):
                prod = (a * b) % p
                e1_val = (prod + c) % p if op1 == PLUS else (prod - c) % p
                e1_dist[i, e1_val] = 1.0

        elif op_or_mul in (PLUS, MINUS):
            # ---------- extended layout: (a +/- b) * c +/- d ----------
            # Before DEDUCT (indices relative to lp):
            # [LP, a, (PLUS/MINUS), b, RP, MUL, c, op2, d, DEDUCT, ...]
            # We'll find MUL position first (robust), then read c, op2, d
            mul_pos_all = (row[:d0] == MUL).nonzero(as_tuple=False)
            if mul_pos_all.numel() == 0:
                continue
            mul = int(mul_pos_all[0].item())

            # Need c at mul+1, op2 at mul+2, d at mul+3, all < d0
            if mul + 3 >= d0:
                continue
            c = int(row[mul + 1].item())
            op2 = int(row[mul + 2].item())
            d   = int(row[mul + 3].item())

            if not (0 <= c < p and op2 in (PLUS, MINUS) and 0 <= d < p):
                continue

            first_partial = (a + b) % p if op_or_mul == PLUS else (a - b) % p
            prod = (first_partial * c) % p
            e1_val = (prod + d) % p if op2 == PLUS else (prod - d) % p
            e1_dist[i, e1_val] = 1.0

        # else: unknown layout -> keep zeros

    return e1_dist, e2_dist





#data generation without parenthesis and end symbol
def generate_noisy_twosteps_dataset_modp_1_1(
    N: int,
    p: int,
    p1: float,
    p2: float,
    *,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    e2_noise_indices: Optional[Sequence[int]] = None,  # kept for API compatibility; ignored (we only corrupt product)
    extended_expression: bool = False,
) -> Tuple[torch.Tensor, "ModularVocabulary", Dict[str, object]]:
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be probabilities in [0, 1].")

    vocab = build_modular_vocabulary(p)

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)

    device = torch.device(device)

    PAD_ID    = vocab["PAD"]
    PLUS_ID   = vocab["PLUS"]
    MINUS_ID  = vocab["MINUS"]
    MUL_ID    = vocab["MUL"]
    DEDUCT_ID = vocab["DEDUCT"]
    # LP / RP / END are intentionally not used in this function anymore

    # Sequence lengths and solution start indices depend on layout
    if extended_expression:
        # After removing LP, RP, END:
        # [a, first_op, b, MUL, c, second_op, d, DEDUCT, product, second_op, d, DEDUCT, result]
        base_len = 13
        solution_start_index = 8   # index of product (first token of E2)
        product_idx_in_e2 = 8      # product position inside E2 for extended layout
        # a is at idx 0, b is at idx 2
        e1_noise_candidates = (0, 2)
    else:
        # After removing LP, RP, END:
        # [a, MUL, b, op, c, DEDUCT, product, op, c, DEDUCT, result]
        base_len = 11
        solution_start_index = 6   # index of product (first token of E2)
        product_idx_in_e2 = 6      # product position inside E2 for short layout
        # a is at idx 0, b is at idx 2
        e1_noise_candidates = (0, 2)

    seqs = torch.full((N, base_len), PAD_ID, dtype=torch.long, device=device)

    for i in range(N):
        if extended_expression:
            # Sample operands
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            d = rng.randint(0, p - 1)
            first_op  = PLUS_ID if rng.random() < 0.5 else MINUS_ID
            second_op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # Compute exact (noise-free) intermediates / result
            first_partial = (a + b) % p if first_op == PLUS_ID else (a - b) % p
            product = (first_partial * c) % p
            result  = (product + d) % p if second_op == PLUS_ID else (product - d) % p

            # Build token sequence WITHOUT LP / RP / END
            tokens: List[int] = [
                a, first_op, b,          # 0,1,2
                MUL_ID, c,               # 3,4
                second_op, d,            # 5,6
                DEDUCT_ID,               # 7
                product,                 # 8  ← product / E2 start
                second_op, d,            # 9,10
                DEDUCT_ID,               # 11
                result,                  # 12
            ]

            # E1 noise: corrupt either a or b (not c/d)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {0, 2}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        else:
            # Short layout: (a * b) +/- c
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            product = (a * b) % p
            result  = (product + c) % p if op == PLUS_ID else (product - c) % p

            # Build token sequence WITHOUT LP / RP / END
            tokens = [
                a, MUL_ID, b,           # 0,1,2
                op, c,                  # 3,4
                DEDUCT_ID,              # 5
                product,                # 6  ← product / E2 start
                op, c,                  # 7,8
                DEDUCT_ID,              # 9
                result,                 # 10
            ]

            # E1 noise: corrupt either a or b (not c)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {0, 2}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        # E2 noise: ONLY corrupt the product token
        if rng.random() < p2:
            tokens[product_idx_in_e2] = _sample_different(
                rng, 0, p - 1, exclude=tokens[product_idx_in_e2]
            )

        # Write sequence
        seqs[i] = torch.tensor(tokens, dtype=torch.long, device=device)

    info = {
        "solution_start_ind": [solution_start_index] * N,
        "vocab_size": vocab.size,
        "modulus": p,
        "base_length": base_len,
    }

    return seqs, vocab, info












#vocabulary dictionary without the parenthesis

def build_modular_vocabulary_wop(p: int) -> ModularVocabulary:
    """Create a vocabulary for arithmetic modulo ``p``.

    Parameters
    ----------
    p:
        A prime modulus. The vocabulary contains integer tokens ``0..p-1``
        plus ``PLUS``, ``MINUS``, ``MUL``, ``DEDUCT`` (``-->``),
        ``END`` (``#``), and ``PAD``.
    """

    # _validate_prime(p)

    # Digits 0..p-1
    tokens: Dict[str, int] = {str(i): i for i in range(p)}
    next_id = p

    def _assign(name: str) -> int:
        nonlocal next_id
        tokens[name] = next_id
        next_id += 1
        return tokens[name]

    # Special tokens (no parentheses anymore)
    _assign("PLUS")
    _assign("MINUS")
    _assign("MUL")
    _assign("DEDUCT")
    _assign("END")
    _assign("PAD")

    return ModularVocabulary(modulus=p, tokens=tokens)









def generate_noisy_twosteps_dataset_modp_wop(
    N: int,
    p: int,
    p1: float,
    p2: float,
    *,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    e2_noise_indices: Optional[Sequence[int]] = None,  # kept for API compatibility; ignored (we only corrupt product)
    extended_expression: bool = False,
) -> Tuple[torch.Tensor, "ModularVocabulary", Dict[str, object]]:
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be probabilities in [0, 1].")

    vocab = build_modular_vocabulary_wop(p)

    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)

    device = torch.device(device)

    PAD_ID    = vocab["PAD"]
    PLUS_ID   = vocab["PLUS"]
    MINUS_ID  = vocab["MINUS"]
    MUL_ID    = vocab["MUL"]
    DEDUCT_ID = vocab["DEDUCT"]
    END_ID    = vocab["END"]

    # Sequence lengths and solution start indices depend on layout
    if extended_expression:
        # no parentheses; layout:
        # a, op1, b, MUL, c, op2, d,
        # DEDUCT, product, op2, d,
        # DEDUCT, result, END
        base_len = 14
        solution_start_index = 8   # index of product after the first DEDUCT
        product_idx_in_e2 = 8      # product position inside E2 for extended layout
        e1_noise_candidates = (0, 2)  # a (idx=0) or b (idx=2)
    else:
        # no parentheses; short layout:
        # a, MUL, b, op, c,
        # DEDUCT, product, op, c,
        # DEDUCT, result, END
        base_len = 12
        solution_start_index = 6   # index of product after the first DEDUCT
        product_idx_in_e2 = 6      # product position inside E2 for short layout
        e1_noise_candidates = (0, 2)  # a (idx=0) or b (idx=2)

    seqs = torch.full((N, base_len), PAD_ID, dtype=torch.long, device=device)

    for i in range(N):
        if extended_expression:
            # Sample operands
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            d = rng.randint(0, p - 1)
            first_op  = PLUS_ID if rng.random() < 0.5 else MINUS_ID
            second_op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            # Compute exact (noise-free) intermediates / result
            first_partial = (a + b) % p if first_op == PLUS_ID else (a - b) % p
            product = (first_partial * c) % p
            result  = (product + d) % p if second_op == PLUS_ID else (product - d) % p

            # Build token sequence WITHOUT parentheses
            tokens: List[int] = [
                a, first_op, b,     # 0,1,2
                MUL_ID, c,          # 3,4
                second_op, d,       # 5,6
                DEDUCT_ID,          # 7
                product,            # 8
                second_op, d,       # 9,10
                DEDUCT_ID,          # 11
                result,             # 12
                END_ID,             # 13
            ]

            # E1 noise: corrupt either a or b (not c/d)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {0, 2}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        else:
            # Short layout without parentheses: (a * b) +/- c
            a = rng.randint(0, p - 1)
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            op = PLUS_ID if rng.random() < 0.5 else MINUS_ID

            product = (a * b) % p
            result  = (product + c) % p if op == PLUS_ID else (product - c) % p

            tokens = [
                a, MUL_ID, b,   # 0,1,2
                op, c,          # 3,4
                DEDUCT_ID,      # 5
                product,        # 6
                op, c,          # 7,8
                DEDUCT_ID,      # 9
                result,         # 10
                END_ID,         # 11
            ]

            # E1 noise: corrupt either a or b (not c)
            if rng.random() < p1:
                idx = rng.choice(e1_noise_candidates)  # choose from {0, 2}
                tokens[idx] = _sample_different(rng, 0, p - 1, exclude=tokens[idx])

        # E2 noise: ONLY corrupt the product token
        if rng.random() < p2:
            tokens[product_idx_in_e2] = _sample_different(
                rng, 0, p - 1, exclude=tokens[product_idx_in_e2]
            )

        # Write sequence
        seqs[i] = torch.tensor(tokens, dtype=torch.long, device=device)

    info = {
        "solution_start_ind": [solution_start_index] * N,
        "vocab_size": vocab.size,
        "modulus": p,
        "base_length": base_len,
    }

    return seqs, vocab, info