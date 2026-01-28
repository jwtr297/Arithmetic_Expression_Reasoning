import torch
import torch.nn.functional as F
from torch.autograd import grad
from typing import Optional, Tuple, Dict, Sequence, List
import torch
import torch.nn as nn
import copy



@torch.no_grad()
def autoreg_generate_from_firstded_to_stop(
    model,
    data: torch.Tensor,                   # [N, L] int64 tokens
    device: Optional[torch.device] = None,
    decode_strategy: str = "greedy",      # "greedy" or "sample"
    temperature: float = 1.0,             # used when decode_strategy == "sample"
    pad_token_id: int = 18,               # padding token
    stop_token_id: int = 17,              # stop token
    ded_token_id: int = 16
) -> torch.Tensor:

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    # work on-device; preserve model mode
    x = data.to(device, non_blocking=True).clone()
    was_training = model.training
    model.eval()

    # per-sample generation (clear & robust)
    for i in range(N):
        seq = x[i].clone()

        # find first non-padding index
        nonpad = (seq != pad_token_id).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            # all padding, nothing to do
            x[i] = seq
            continue
        start_idx = int(nonpad[0].item())

        # find first derivation symbol  at/after start_idx
        idx_ded = (seq == ded_token_id).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded.numel() == 0:
            x[i] = seq
            continue
        cand = idx_ded[idx_ded >= start_idx]
        if cand.numel() == 0:
            x[i] = seq
            continue
        first_ded = int(cand[0].item())

        gen_seq = seq.clone()
        pos = first_ded + 1
        while pos < L:
            # safety: mask future positions to PAD to avoid leakage
            inp = gen_seq.clone()
            if pos + 1 < L:
                inp[pos+1:] = pad_token_id

            # forward
            logits, _ = model(inp.unsqueeze(0))        # [1, L, V]
            # assume logits[t] predicts token at t+1; so to predict at `pos`, read logits at pos-1
            next_logits = logits[0, pos - 1, :]        # [V]

            # decode
            if decode_strategy == "greedy":
                nxt = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            gen_seq[pos] = nxt

            # if STOP is produced, pad the rest and break
            if nxt == stop_token_id:
                if pos + 1 < L:
                    gen_seq[pos+1:] = pad_token_id
                break

            pos += 1

        x[i] = gen_seq

    if was_training:
        model.train()

    return x



def eval_expr_modp(tokens: torch.Tensor, p: int = 11) -> Optional[int]:
    
    
    PLUS_ID  = p
    MINUS_ID = p+1
    LP_ID    = p+2
    RP_ID    = p+3
    MUL_ID   = p+4 
    SEP_ID   = p+5
    EOS_ID   = p+6
    PAD_ID   = p+7
    VOCAB_SIZE = p+8

    
    if tokens.numel() == 0:
        return None

    xs = tokens.tolist()
    # precedence: '*' > '+' == '-'
    prec = {PLUS_ID:1, MINUS_ID:1, MUL_ID:2}

    def is_digit(t): return 0 <= t <= p-1
    def is_op(t):    return t in (PLUS_ID, MINUS_ID, MUL_ID)

    # infix -> RPN
    out, st = [], []
    for t in xs:
        if is_digit(t):
            out.append(t)
        elif t == LP_ID:
            st.append(t)
        elif t == RP_ID:
            while st and st[-1] != LP_ID:
                out.append(st.pop())
            if not st or st[-1] != LP_ID:
                return None  # mismatched ')'
            st.pop()  # pop '('
        elif is_op(t):
            while st and st[-1] in prec and prec[st[-1]] >= prec[t]:
                out.append(st.pop())
            st.append(t)
        else:
            # any other token makes expression invalid
            return None

    while st:
        if st[-1] in (LP_ID, RP_ID):
            return None
        out.append(st.pop())

    # evaluate RPN mod p
    stk = []
    for t in out:
        if is_digit(t):
            stk.append(t % p)
        else:
            if len(stk) < 2:
                return None
            b = stk.pop()
            a = stk.pop()
            if   t == PLUS_ID:  stk.append((a + b) % p)
            elif t == MINUS_ID: stk.append((a - b) % p)
            elif t == MUL_ID:   stk.append((a * b) % p)
            else:               return None
    if len(stk) != 1:
        return None
    return int(stk[0] % p)






def proportion_E1_eq_r_but_E2_neq_r(
    seqs: torch.Tensor, p: int = 11
) -> Tuple[float, torch.Tensor]:


    PLUS_ID  = p
    MINUS_ID = p+1
    LP_ID    = p+2
    RP_ID    = p+3
    MUL_ID   = p+4
    SEP_ID   = p+5
    EOS_ID   = p+6
    PAD_ID   = p+7
    VOCAB_SIZE = p + 8

    
    assert seqs.ndim == 2 and seqs.size(1) == 48 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        # first non-PAD index
        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        # find all SEP(16) positions
        seps = (row == SEP_ID).nonzero(as_tuple=False).squeeze(-1)
        if seps.numel() < 2:
            continue
        # first SEP after start, then the next one
        first_seps = seps[seps >= start]
        if first_seps.numel() == 0:
            continue
        first_sep = int(first_seps[0].item())
        after_first = seps[seps > first_sep]
        if after_first.numel() == 0:
            continue
        second_sep = int(after_first[0].item())

        # r should be the token right after the second SEP
        if second_sep + 1 >= L:
            continue
        r_tok = int(row[second_sep + 1].item())
        if not (0 <= r_tok <= 10):
            continue  # r must be a digit

        # E1 tokens: [start, first_sep)
        e1_tokens = row[start:first_sep]
        # E2 tokens: (first_sep, second_sep)
        if second_sep - first_sep <= 1:
            continue
        e2_tokens = row[first_sep + 1: second_sep]

        # evaluate under mod p
        e1_val = eval_expr_modp(e1_tokens, p=p)
        e2_val = eval_expr_modp(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        # this sample is valid for denominator
        denom_mask[i] = True
        count_den += 1

        # numerator condition: E1 == r AND E2 != r
        if (e1_val == r_tok) and (e2_val != r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio








#intervention metric: E2-intervention
@torch.no_grad()
def measure_e2_all_digits_intervention_ce_modp(
    model,
    data: torch.Tensor,                       # [N, L] int64 tokens
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of random full-E2 replacements per sample
    digits_for_replace: Optional[Sequence[int]] = None,  # if None, defaults to all digits except {0,5} (when 5 < p)
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Measure intervention cross-entropy and invariance when replacing ALL digit tokens in E2,
    generalized to modulo-p tokenization.

    Token IDs under mod p:
      digits:         0 .. p-1
      plus:           p
      minus:          p+1
      left parenthesis:  p+2
      right parenthesis: p+3
      multiply:       p+4
      derivation (DED/>>> marker): p+5       # used to delimit E2 as [first DED, second DED]
      stop:           p+6
      pad:            p+7

    Returns:
      (avg_ce_all, per_sample_avg_ce_cpu, inv_mean, per_sample_inv_avg_cpu)
      - avg_ce_all: scalar float, average cross-entropy under interventions (P vs Q)
      - per_sample_avg_ce_cpu: [N] tensor of per-source-sample average CE
      - inv_mean: scalar float, average invariance rate (argmax preserved)
      - per_sample_inv_avg_cpu: [N] tensor of per-source-sample average invariance
    """

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Derived special token IDs for mod p
    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # Helper: extract logits for E3 position and keep only digit classes [0..p-1]
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Assumes E3 is located at -3 (third from last). Adjust if your layout differs.
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # 1) Forward original sequences to get baseline digit distribution P over E3
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)  # per-sample baseline probs over digits
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]

    # 2) Build intervention variants by replacing ALL digit tokens in E2
    modified_seqs: list[torch.Tensor] = []
    modified_src_idx: list[int] = []

    # Default replacement digits: all digits except {0, 5} if 5 exists; else exclude only 0
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]  # [L], still on CPU (cloned later to device when batching)

        # Find first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # Find two DED markers (first and second) at/after start_idx
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() < 2:
            continue

        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = idx_ded_all[idx_ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # E2 slice must have at least one token between the two DEDs
        if second_ded <= first_ded + 1:
            continue

        e2_slice = seq[first_ded + 1 : second_ded]
        # Positions in E2 that are digit tokens (0..p-1)
        rel_digits = (e2_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digits.numel() == 0:
            continue

        # Absolute positions of those digits in the sequence
        e2_digit_pos = rel_digits + (first_ded + 1)

        # Create num_variants interventions per sample
        for _ in range(num_variants):
            new_seq = seq.clone()
            skip_variant = False

            #Replace every digit position in E2 with a random allowed digit (not equal to original if possible)
            for pos_t in e2_digit_pos:
                pos = int(pos_t.item())
                orig = int(new_seq[pos].item())

                if orig in allowed_digits and len(allowed_digits) > 1:
                    # Exclude original to ensure a change when possible
                    candidates = [d for d in allowed_digits if d != orig]
                else:
                    candidates = allowed_digits

                if not candidates:
                    skip_variant = True
                    break

                new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
                new_seq[pos] = new_digit

            if skip_variant:
                continue

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # If no interventions could be formed, return zeros
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    # Stack all modified sequences and map to their source indices
    modified_tensor = torch.stack(modified_seqs, dim=0)                   # [M, L] on CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)  # [M] on device
    M = modified_tensor.shape[0]

    total_ce = 0.0    # sum of cross-entropies over all interventions
    total_inv = 0.0   # sum of invariance indicators over all interventions
    count = 0         # number of interventions

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    # 3) Forward all modified sequences to get Q and compute CE(P || Q) and invariance
    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p], numeric stability

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]

            # Cross-entropy between baseline P and intervened Q: CE(P, Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # Invariance indicator: whether argmax digit under P equals argmax under Q
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    # Return per-sample tensors on CPU for convenient logging/serialization
    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()





#intervention metric: E1-intervention
@torch.no_grad()
def measure_e1_ab_digits_intervention_ce_modp(  # mod-p version: intervene only on a and b
    model,
    data: torch.Tensor,                       # [N, L] int64 tokens
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of variants per sample
    digits_for_replace: Optional[Sequence[int]] = None,  # if None, defaults below
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    For each sample, create `num_variants` modified copies by replacing
    ONLY the digit tokens corresponding to `a` and `b` inside E1.

    We identify `a` and `b` as the FIRST TWO digit tokens (0..p-1) in E1,
    consistent with the short layout of `generate_noisy_twosteps_dataset_modp_1`:

        (a * b) +/- c

    Token IDs under mod p:
      digits:            0 .. p-1
      plus:              p
      minus:             p+1
      left parenthesis:  p+2
      right parenthesis: p+3
      multiply:          p+4
      derivation (DED):  p+5            # E1 ends right before the first DED at/after start
      stop:              p+6
      pad:               p+7

    Definitions:
      E1 = [first_non_PAD_index, first_DED_index)

    Metrics:
      - Cross-entropy CE(P,Q): P = original E3 distribution, Q = modified E3 distribution
      - MAP invariance: 1[ argmax(P) == argmax(Q) ]
    E3 readout: logits[:, -3, :][:, :p]

    Returns:
        avg_ce_all (float): mean CE over all modified variants
        per_sample_avg_ce [N]: mean CE per original sample (0 if no variants)
        inv_mean (float): global mean of MAP invariance over all variants
        per_sample_inv_avg [N]: mean MAP invariance per original sample (0 if no variants)
    """

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Special token ids for mod p
    DED_ID = p + 5
    PAD_ID = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # Helper: extract E3 logits and keep only digit classes [0..p-1]
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Assumes E3 is located at -3 (third-from-last). Adjust if your layout differs.
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # 1) Baseline distributions P over digits at E3 for all original sequences
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                # [B, L, V]
            e3_logits = _e3_logits_from(logits)     # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)   # [B, p]

    # 2) Create intervention variants by replacing ONLY a and b (first two digits in E1)
    modified_seqs: list[torch.Tensor] = []
    modified_src_idx: list[int] = []

    # Default replacement set:
    #  - if p > 5: all digits except {0, 5}
    #  - else: all digits except {0}
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]  # [L] (CPU tensor)

        # (a) first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # (b) first DED at/after start_idx
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() == 0:
            continue
        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        # E1 = [start_idx, first_ded)
        if first_ded <= start_idx:
            continue
        e1_slice = seq[start_idx:first_ded]

        # positions of digit tokens (0..p-1) in E1
        rel_digits = (e1_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digits.numel() < 2:
            # Need at least a and b
            continue

        # take the first two digit positions -> interpreted as a, b
        rel_ab = rel_digits[:2]
        e1_ab_pos = rel_ab + start_idx  # absolute positions in seq (len=2)

        # build num_variants variants
        for _ in range(num_variants):
            new_seq = seq.clone()
            skip_variant = False

            for pos_t in e1_ab_pos:
                pos = int(pos_t.item())
                orig = int(new_seq[pos].item())

                # Prefer changing the digit if possible
                if orig in allowed_digits and len(allowed_digits) > 1:
                    candidates = [d for d in allowed_digits if d != orig]
                else:
                    candidates = allowed_digits

                if not candidates:
                    skip_variant = True
                    break

                new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
                new_seq[pos] = new_digit

            if skip_variant:
                continue

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # If no variants could be created, return zeros
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    modified_tensor = torch.stack(modified_seqs, dim=0)  # [M, L] (CPU)
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)  # [M] (device)
    M = modified_tensor.shape[0]

    # 3) Evaluate modified sequences and compute CE(P,Q) and MAP invariance
    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)   # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]

            # CE stability: use log-softmax directly
            logQ = F.log_softmax(e3_logits_mod, dim=-1)                  # [B, p]
            q_argmax = e3_logits_mod.argmax(dim=-1)                      # [B]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]
            p_argmax = P_sel.argmax(dim=-1)                              # [B]

            # Cross-entropy CE(P,Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # MAP invariance: whether argmax digit is preserved
            inv_vec = (p_argmax == q_argmax).float()                     # [B]

            # Global accumulators
            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            # Per-sample accumulators
            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce   = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg  = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    # Return per-sample tensors on CPU for convenient logging/serialization
    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()




# ==============================
# E2 = r
# ==============================

def _eval_expr_modp(tokens: torch.Tensor, p: int) -> Optional[int]:
    """
    Evaluate an arithmetic expression modulo p with token IDs:
      digits 0..p-1, '+'=p, '-'=p+1, '('=p+2, ')'=p+3, '*'=p+4.
    Returns an int in [0, p-1] or None if invalid (syntax error, stack underflow, etc.).
    Uses shunting-yard to convert to RPN, then evaluates mod p.
    """
    if tokens.numel() == 0:
        return None
    xs = tokens.tolist()

    PLUS, MINUS, LPAREN, RPAREN, MUL = p, p+1, p+2, p+3, p+4
    # operator precedence: * > (+,-)
    prec = {PLUS: 1, MINUS: 1, MUL: 2}

    def is_digit(t: int) -> bool:
        return 0 <= t < p

    def is_op(t: int) -> bool:
        return t in (PLUS, MINUS, MUL)

    # 1) shunting-yard -> RPN
    out, st = [], []
    for t in xs:
        if is_digit(t):
            out.append(t)
        elif t == LPAREN:
            st.append(t)
        elif t == RPAREN:
            # pop until LPAREN
            while st and st[-1] != LPAREN:
                out.append(st.pop())
            if not st or st[-1] != LPAREN:
                return None
            st.pop()
        elif is_op(t):
            while st and st[-1] in prec and prec[st[-1]] >= prec[t]:
                out.append(st.pop())
            st.append(t)
        else:
            # unknown token
            return None

    while st:
        if st[-1] in (LPAREN, RPAREN):
            return None
        out.append(st.pop())

    # 2) evaluate RPN mod p
    stk = []
    for t in out:
        if is_digit(t):
            stk.append(t % p)
        else:
            if len(stk) < 2:
                return None
            b = stk.pop(); a = stk.pop()
            if t == PLUS:
                stk.append((a + b) % p)
            elif t == MINUS:
                stk.append((a - b) % p)
            elif t == MUL:
                stk.append((a * b) % p)
            else:
                return None
    if len(stk) != 1:
        return None
    return int(stk[0])


# ---- main: compute ratio under mod p ----
@torch.no_grad()
def e2_modp_vs_model_ratio(
    model,
    data: torch.Tensor,                 # [N, L] int64 tokens
    p: int,                             # modulus p (digits are 0..p-1)
    device: Optional[torch.device] = None,
    eval_batch_size: int = 512,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    For each sample:
      - find the first and second derivation markers (DED) = p+5
        Define E2 = (firstDED, secondDED)  (exclusive of both ends)
      - compute E2 value via the parser above (mod p)
      - run the model and take logits at position = secondDED
        (interpreted as predicting token at secondDED+1).
        Restrict logits to digit classes [0..p-1] and take argmax as the prediction.
      - compare prediction with parsed E2 value.

    Returns:
      ratio (float): fraction of valid samples where pred == parsed E2 (mod p)
      details (dict of CPU tensors):
        - 'valid_mask' : [N] bool (E2 parsed & DEDs exist)
        - 'e2_modp'    : [N] int64 (=-1 if invalid)
        - 'pred_digit' : [N] int64 (=-1 if invalid)
        - 'match'      : [N] bool (only meaningful where valid_mask=True)

    Notes:
      * If your model's readout position or class layout differs, adjust the
        gathering of logits accordingly (marked below).
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Special token IDs under mod p
    DED_ID = p + 5
    PAD_ID = p + 7

    # resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    x = data.to(device, non_blocking=True)
    was_training = model.training
    model.eval()

    # Precompute per-sample: first & second DED and parsed E2 value (mod p)
    first_ded_idx  = torch.full((N,), -1, dtype=torch.long, device=device)
    second_ded_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    e2_modp        = torch.full((N,), -1, dtype=torch.long, device=device)
    valid_mask     = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(N):
        seq = x[i]

        # first non-PAD index
        nonpad = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start_idx = int(nonpad[0].item())

        # find DED markers
        idx_ded = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded.numel() < 2:
            continue

        # first DED at/after start
        cand1 = idx_ded[idx_ded >= start_idx]
        if cand1.numel() == 0:
            continue
        f_ded = int(cand1[0].item())

        # second DED strictly after first
        cand2 = idx_ded[idx_ded > f_ded]
        if cand2.numel() == 0:
            continue
        s_ded = int(cand2[0].item())

        if s_ded <= f_ded + 1:
            continue

        e2_tokens = seq[f_ded + 1 : s_ded]
        val = _eval_expr_modp(e2_tokens, p)
        if val is None:
            continue

        first_ded_idx[i]  = f_ded
        second_ded_idx[i] = s_ded
        e2_modp[i]        = val
        valid_mask[i]     = True

    # If no valid samples, return zeros
    if valid_mask.sum().item() == 0:
        details = {
            "valid_mask": valid_mask.cpu(),
            "e2_modp": e2_modp.cpu(),
            "pred_digit": torch.full((N,), -1, dtype=torch.long),
            "match": torch.zeros(N, dtype=torch.bool),
        }
        if was_training:
            model.train()
        return 0.0, details

    # Run model; for each sample, read logits at position = second_ded (predict next token)
    pred_digit = torch.full((N,), -1, dtype=torch.long, device=device)
    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = x[s:e]                                  # [B, L]
        logits, _ = model(batch)                        # [B, L, V]
        # positions inside the batch
        pos = second_ded_idx[s:e].clone()               # [B]
        valid_b = (pos >= 0)
        if valid_b.any():
            # gather logits at those positions, restrict to digit classes [0..p-1]
            idx = torch.arange(e - s, device=device)
            gathered = logits[idx, pos.clamp_min(0), :][:, :p]  # [B, p]
            pd = torch.argmax(gathered, dim=-1)
            pred_digit[s:e][valid_b] = pd[valid_b]

    # Compare only on valid rows
    match = (pred_digit == e2_modp) & valid_mask
    ratio = (match.sum().float() / valid_mask.sum().float()).item()

    if was_training:
        model.train()

    details = {
        "valid_mask": valid_mask.detach().cpu(),
        "e2_modp": e2_modp.detach().cpu(),
        "pred_digit": pred_digit.detach().cpu(),
        "match": match.detach().cpu(),
    }
    return ratio, details






# ==============================
# E1 = r
# ==============================
def _eval_expr_modp(tokens: torch.Tensor, p: int) -> Optional[int]:
    """
    Evaluate an arithmetic expression modulo p from token IDs:
      digits 0..p-1, '+'=p, '-'=p+1, '('=p+2, ')'=p+3, '*'=p+4.
    Returns:
      - int in [0, p-1] if the expression is valid
      - None if invalid (e.g., unbalanced parentheses or malformed expression)
    Uses shunting-yard to convert to RPN, then evaluates the RPN mod p.
    """
    if tokens.numel() == 0:
        return None
    xs = tokens.tolist()

    PLUS, MINUS, LPAREN, RPAREN, MUL = p, p+1, p+2, p+3, p+4
    prec = {PLUS: 1, MINUS: 1, MUL: 2}  # operator precedence

    def is_digit(t: int) -> bool: return 0 <= t < p
    def is_op(t: int) -> bool:    return t in (PLUS, MINUS, MUL)

    # To Reverse Polish Notation (RPN)
    out, st = [], []
    for t in xs:
        if is_digit(t):
            out.append(t)
        elif t == LPAREN:
            st.append(t)
        elif t == RPAREN:
            while st and st[-1] != LPAREN:
                out.append(st.pop())
            if not st or st[-1] != LPAREN:
                return None
            st.pop()
        elif is_op(t):
            while st and st[-1] in prec and prec[st[-1]] >= prec[t]:
                out.append(st.pop())
            st.append(t)
        else:
            return None

    while st:
        if st[-1] in (LPAREN, RPAREN):
            return None
        out.append(st.pop())

    # Evaluate RPN mod p
    stk = []
    for t in out:
        if is_digit(t):
            stk.append(t % p)
        else:
            if len(stk) < 2:
                return None
            b = stk.pop(); a = stk.pop()
            if   t == PLUS:  stk.append((a + b) % p)
            elif t == MINUS: stk.append((a - b) % p)
            elif t == MUL:   stk.append((a * b) % p)
            else: return None

    if len(stk) != 1:
        return None
    return int(stk[0])


# ---------- main: compare E1 mod-p vs model prediction after second DED ----------
@torch.no_grad()
def e1_modp_vs_model_ratio(
    model,
    data: torch.Tensor,                 # [N, L] int64 tokens
    p: int,                             # modulus p (digits are 0..p-1)
    device: Optional[torch.device] = None,
    eval_batch_size: int = 512,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Special token IDs under mod p
    DED_ID = p + 5
    PAD_ID = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    x = data.to(device, non_blocking=True)
    was_training = model.training
    model.eval()

    # Precompute indices and E1 values per sample
    first_ded_idx  = torch.full((N,), -1, dtype=torch.long, device=device)
    second_ded_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    e1_modp        = torch.full((N,), -1, dtype=torch.long, device=device)
    valid_mask     = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(N):
        seq = x[i]

        # First non-PAD token
        nonpad = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start_idx = int(nonpad[0].item())

        # Locate first and second DED
        idx_ded = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded.numel() < 2:
            continue

        cand1 = idx_ded[idx_ded >= start_idx]
        if cand1.numel() == 0:
            continue
        f_ded = int(cand1[0].item())

        cand2 = idx_ded[idx_ded > f_ded]
        if cand2.numel() == 0:
            continue
        s_ded = int(cand2[0].item())

        # Build E1 and evaluate mod p
        if f_ded <= start_idx:
            continue
        e1_tokens = seq[start_idx:f_ded]
        val = _eval_expr_modp(e1_tokens, p)
        if val is None:
            continue

        first_ded_idx[i]  = f_ded
        second_ded_idx[i] = s_ded
        e1_modp[i]        = val
        valid_mask[i]     = True

    # Early exit if no valid samples
    if valid_mask.sum().item() == 0:
        details = {
            "valid_mask": valid_mask.cpu(),
            "e1_modp": e1_modp.cpu(),
            "pred_digit": torch.full((N,), -1, dtype=torch.long),
            "match": torch.zeros(N, dtype=torch.bool),
        }
        if was_training:
            model.train()
        return 0.0, details

    # Read model prediction for token (second_ded + 1)
    pred_digit = torch.full((N,), -1, dtype=torch.long, device=device)
    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = x[s:e]                      # [B, L]
        logits, _ = model(batch)            # [B, L, V]
        pos = second_ded_idx[s:e].clone()   # [B]
        valid_b = (pos >= 0)
        if valid_b.any():
            idx = torch.arange(e - s, device=device)
            # Restrict to digit classes [0..p-1]; adjust this slice if your vocab differs.
            gathered = logits[idx, pos.clamp_min(0), :][:, :p]  # [B, p]
            pd = torch.argmax(gathered, dim=-1)
            pred_digit[s:e][valid_b] = pd[valid_b]

    # Compute ratio on valid rows only
    match = (pred_digit == e1_modp) & valid_mask
    ratio = (match.sum().float() / valid_mask.sum().float()).item()

    if was_training:
        model.train()

    details = {
        "valid_mask":  valid_mask.detach().cpu(),
        "e1_modp":    e1_modp.detach().cpu(),
        "pred_digit":  pred_digit.detach().cpu(),
        "match":       match.detach().cpu(),
    }
    return ratio, details










@torch.no_grad()
def measure_e2_all_digits_intervention_ce_modp_short(
    model,
    data: torch.Tensor,                       
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of random replacements per sample
    digits_for_replace: Optional[Sequence[int]] = None,  # allowed digits for product replacement
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    seed: Optional[int] = None, 
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Evaluate robustness for SHORT layout by intervening ONLY the E2 product token.

    Differences from the "all-digits-in-E2" version:
      - We locate E2 as the span between the first and second DED markers.
      - Inside E2, we identify the *product* as the FIRST digit token in that span,
        and we only replace that single token (one-point intervention).
      - E3 (final result) for short layout is assumed at index -2 (just before END).
    """

    # Basic checks
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Special token IDs for mod p
    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # --- Short layout: final result (E3) is at -2 (just before END) ---
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Restrict to digit classes [0..p-1]
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # 1) Forward originals to get baseline distribution P over digits at E3
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]

    # 2) Build intervention variants by replacing ONLY the E2 product token
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    # Allowed digits for replacement
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]  # [L], CPU tensor

        # Find first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # Locate the two DED markers delimiting E2 in short layout
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() < 2:
            continue

        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = idx_ded_all[idx_ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # E2 must contain at least one token
        if second_ded <= first_ded + 1:
            continue

        e2_slice = seq[first_ded + 1 : second_ded]

        # Identify the product as the FIRST digit token in E2
        rel_digit_positions = (e2_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digit_positions.numel() == 0:
            # No digit in E2; skip
            continue
        first_rel_digit = int(rel_digit_positions[0].item())
        product_abs_pos = first_ded + 1 + first_rel_digit

        # Create num_variants interventions per sample (replace ONLY the product token)
        orig_product = int(seq[product_abs_pos].item())
        for _ in range(num_variants):
            new_seq = seq.clone()

            # Build candidate set (exclude original when possible)
            if orig_product in allowed_digits and len(allowed_digits) > 1:
                candidates = [d for d in allowed_digits if d != orig_product]
            else:
                candidates = allowed_digits

            if not candidates:
                continue

            new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
            new_seq[product_abs_pos] = new_digit

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # If no interventions were possible, return zeros
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    # Stack all modified sequences and map to their source indices
    modified_tensor = torch.stack(modified_seqs, dim=0)                                   # [M, L] CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)      # [M] on device
    M = modified_tensor.shape[0]

    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    # 3) Forward modified sequences to get Q and compute CE(P || Q) and invariance
    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]

            # Cross-entropy CE(P, Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # Invariance: argmax preserved?
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    # Return per-sample tensors on CPU
    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()






@torch.no_grad()
def measure_e2_sec_digits_intervention_ce_modp_short(
    model,
    data: torch.Tensor,                       
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of random replacements per sample
    digits_for_replace: Optional[Sequence[int]] = None,  # allowed digits for product replacement
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    seed: Optional[int] = None, 
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Evaluate robustness for SHORT layout by intervening ONLY the E2 product token.

    Differences from the "all-digits-in-E2" version:
      - We locate E2 as the span between the first and second DED markers.
      - Inside E2, we identify the *product* as the FIRST digit token in that span,
        and we only replace that single token (one-point intervention).
      - E3 (final result) for short layout is assumed at index -2 (just before END).
    """

    # Basic checks
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Special token IDs for mod p
    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # --- Short layout: final result (E3) is at -2 (just before END) ---
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Restrict to digit classes [0..p-1]
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # 1) Forward originals to get baseline distribution P over digits at E3
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]

    # 2) Build intervention variants by replacing ONLY the E2 product token
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    # Allowed digits for replacement
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]  # [L], CPU tensor

        # Find first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # Locate the two DED markers delimiting E2 in short layout
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() < 2:
            continue

        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = idx_ded_all[idx_ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # E2 must contain at least one token
        if second_ded <= first_ded + 1:
            continue

        e2_slice = seq[first_ded + 1 : second_ded]

        # Identify the product as the SECOND digit token in E2
        rel_digit_positions = (e2_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digit_positions.numel() == 0:
            # No digit in E2; skip
            continue
        first_rel_digit = int(rel_digit_positions[1].item())
        product_abs_pos = first_ded + 1 + first_rel_digit

        # Create num_variants interventions per sample (replace ONLY the product token)
        orig_product = int(seq[product_abs_pos].item())
        for _ in range(num_variants):
            new_seq = seq.clone()

            # Build candidate set (exclude original when possible)
            if orig_product in allowed_digits and len(allowed_digits) > 1:
                candidates = [d for d in allowed_digits if d != orig_product]
            else:
                candidates = allowed_digits

            if not candidates:
                continue

            new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
            new_seq[product_abs_pos] = new_digit

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # If no interventions were possible, return zeros
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    # Stack all modified sequences and map to their source indices
    modified_tensor = torch.stack(modified_seqs, dim=0)                                   # [M, L] CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)      # [M] on device
    M = modified_tensor.shape[0]

    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    # 3) Forward modified sequences to get Q and compute CE(P || Q) and invariance
    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]

            # Cross-entropy CE(P, Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # Invariance: argmax preserved?
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    # Return per-sample tensors on CPU
    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()





@torch.no_grad()
def measure_e1_ab_digits_intervention_ce_modp_short(
    model,
    data: torch.Tensor,                       # [N, L] int64 tokens
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of variants per sample
    digits_for_replace: Optional[Sequence[int]] = None,
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Intervention metric for short-form (a * b) ± c expressions:
      - Only intervene on 'a' and 'b' tokens in E1.

    Token IDs under mod p:
      digits: 0 .. p-1
      plus:   p
      minus:  p+1
      lp:     p+2
      rp:     p+3
      mul:    p+4
      ded:    p+5
      stop:   p+6
      pad:    p+7
    """

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Token IDs
    DED_ID = p + 5
    PAD_ID = p + 7

    # Device resolution
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # Helper: extract E3 logits
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # === 1) Get baseline P ===
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)
            e3_logits = _e3_logits_from(logits)
            P[s:e] = F.softmax(e3_logits, dim=-1)

    # === 2) Build intervention variants (only a, b) ===
    modified_seqs = []
    modified_src_idx = []

    # Allowed digits
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]

        # Find start and first DED
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        ded_idx_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if ded_idx_all.numel() == 0:
            continue
        first_ded = int(ded_idx_all[ded_idx_all >= start_idx][0].item())

        # E1 slice
        e1_slice = seq[start_idx:first_ded]
        rel_digits = (e1_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digits.numel() < 2:
            continue

        # a, b positions (first two digits)
        rel_ab = rel_digits[:2]
        ab_abs_pos = rel_ab + start_idx

        # Create num_variants
        for _ in range(num_variants):
            new_seq = seq.clone()
            skip_variant = False

            for pos_t in ab_abs_pos:
                pos = int(pos_t.item())
                orig = int(new_seq[pos].item())

                candidates = [d for d in allowed_digits if d != orig] if len(allowed_digits) > 1 else allowed_digits
                if not candidates:
                    skip_variant = True
                    break

                new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
                new_seq[pos] = new_digit

            if skip_variant:
                continue

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # === 3) Compute CE and invariance ===
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zeros = torch.zeros(N)
        return 0.0, zeros, 0.0, zeros

    modified_tensor = torch.stack(modified_seqs, dim=0)
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)
    M = modified_tensor.shape[0]

    total_ce = total_inv = 0.0
    count = 0

    per_sample_sum_ce = torch.zeros(N, device=device)
    per_sample_sum_inv = torch.zeros(N, device=device)
    per_sample_cnt = torch.zeros(N, device=device, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)
            logits_mod, _ = model(batch)
            e3_logits_mod = _e3_logits_from(logits_mod)

            logQ = F.log_softmax(e3_logits_mod, dim=-1)
            q_argmax = e3_logits_mod.argmax(dim=-1)

            ori_idx = src_idx_tensor[s:e]
            P_sel = P[ori_idx]
            p_argmax = P_sel.argmax(dim=-1)

            ce_vec = -(P_sel * logQ).sum(dim=-1)
            inv_vec = (p_argmax == q_argmax).float()

            total_ce += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean = total_inv / max(count, 1)

    per_sample_avg_ce = per_sample_sum_ce / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    return (
        float(avg_ce_all),
        per_sample_avg_ce.detach().cpu(),
        float(inv_mean),
        per_sample_inv_avg.detach().cpu(),
    )












@torch.no_grad()
def compute_e3_perplexity_modp_short(
    model,
    data: torch.Tensor,                  # [N, L] int64 tokens
    p: int,                              # modulus p (digits 0..p-1; specials start at p)
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    # optional overrides if your layout shifts; defaults match short layout + next-token prediction
    e3_logits_pos: int = -3,             # read logits at -3 to predict the E3 token at -2
    e3_target_pos: int = -2,             # ground-truth E3 token position
) -> Tuple[torch.Tensor, float]:
    """
    Compute per-sample perplexity for the E3 token only (short layout),
    aligning with the intervention metric:
      - read logits at index -3 (next-token prediction),
      - restrict to digit classes [0..p-1] before softmax,
      - target is the E3 token at index -2.

    Returns:
      per_sample_ppl: [N] tensor of PPL per sample (CPU)
      ppl_mean: float, mean PPL across samples
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    was_training = model.training
    model.eval()

    per_sample_ppl = torch.empty(N, dtype=torch.float64)

    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = data[s:e].to(device, non_blocking=True)      # [B, L]

        # Forward pass
        out = model(batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out   # [B, L, V]

        # Read E3 logits at -3 and restrict to digit classes [0..p-1]
        e3_logits = logits[:, e3_logits_pos, :][:, :p]       # [B, p]

        # Ground-truth E3 token (digit) at -2
        targets = batch[:, e3_target_pos]                    # [B], values in 0..p-1

        # Log-prob over digits only (renormalized inside 0..p-1)
        log_probs_digits = F.log_softmax(e3_logits, dim=-1)  # [B, p]
        tgt_logp = log_probs_digits.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # Per-sample perplexity for a single position = exp(NLL)
        ppl = torch.exp(-tgt_logp).double().cpu()            # [B]
        per_sample_ppl[s:e] = ppl

    ppl_mean = float(per_sample_ppl.mean().item())

    if was_training:
        model.train()

    return per_sample_ppl, ppl_mean






def _prior_product_modp(p: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    Prior over true product t = (a*b) mod p with a,b ~ Unif{0..p-1}:
      pi[0] = (2p-1)/p^2;  pi[t!=0] = (p-1)/p^2
    """
    pi = torch.full((p,), (p - 1) / (p**2), dtype=dtype, device=device)
    pi[0] = (2 * p - 1) / (p**2)
    return pi




@torch.no_grad()
def exact_E3_entropy_sharpness_from_data_short(
    data: torch.Tensor,         # [N, L] int64 tokens (short layout)
    p: int,                     # modulus p (digits 0..p-1)
    p2: torch.Tensor | float,   # scalar or [N] tensor: corruption prob for product in E2
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tensorized, sample-free, closed-form computation for SHORT layout.
    From the input sequences, parse E2=(product, op, c), then compute exact:
      - P(E3 | E2)  [N, p]
      - H           [N]   (Shannon entropy over digits)
      - sharp=exp(H)[N]   (entropy-based sharpness)
      - H_norm=H/log(p) [N]
      - valid_mask   [N]   (rows where E2 was parsed and op is +/-)

    Notes:
      - Uses DED positions to locate E2 span. Robust to PAD.
      - Only depends on (p, p2, parsed E2); does NOT use model outputs.
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    device = data.device
    dtype  = torch.float64

    # Special token ids
    PLUS_ID  = p
    MINUS_ID = p + 1
    DED_ID   = p + 5
    PAD_ID   = p + 7

    # p2 to tensor [N]
    if isinstance(p2, (float, int)):
        p2 = torch.tensor(p2, device=device, dtype=dtype)
    p2 = p2.to(device=device, dtype=dtype)
    if p2.dim() == 0:
        p2 = p2.expand(N)  # [N]

    # ----- locate first non-PAD, first DED, second DED (vectorized) -----
    idx = torch.arange(L, device=device).view(1, -1).expand(N, -1)  # [N, L]

    # first non-PAD index
    nonpad_mask = (data != PAD_ID)
    big = torch.full((N, L), L + 10, device=device, dtype=torch.long)
    masked_idx = torch.where(nonpad_mask, idx, big)
    start_idx = masked_idx.min(dim=1).values  # [N], >=0 if any non-PAD

    # first DED index
    ded_mask = (data == DED_ID)
    masked_ded = torch.where(ded_mask, idx, big)
    first_ded = masked_ded.min(dim=1).values  # [N], big if none

    # second DED index: min idx > first_ded
    # mask out positions <= first_ded
    gt_first = idx > first_ded.view(-1, 1)
    masked_ded2 = torch.where(ded_mask & gt_first, idx, big)
    second_ded = masked_ded2.min(dim=1).values  # [N]

    # E2 must have at least 3 tokens: product, op, c
    valid_span = (first_ded < big.max()) & (second_ded < big.max()) & (start_idx < first_ded) & (second_ded >= first_ded + 3)

    # ----- parse E2 tokens (product, op, c) -----
    row = torch.arange(N, device=device)
    prod_pos = (first_ded + 1).clamp_max(L - 1)
    op_pos   = (first_ded + 2).clamp_max(L - 1)
    c_pos    = (first_ded + 3).clamp_max(L - 1)

    product_obs = torch.zeros(N, dtype=torch.long, device=device)
    op_id       = torch.zeros(N, dtype=torch.long, device=device)
    c_val       = torch.zeros(N, dtype=torch.long, device=device)

    product_obs[valid_span] = data[row[valid_span], prod_pos[valid_span]]
    op_id[valid_span]       = data[row[valid_span], op_pos[valid_span]]
    c_val[valid_span]       = data[row[valid_span], c_pos[valid_span]]

    op_valid = (op_id == PLUS_ID) | (op_id == MINUS_ID)
    valid_mask = valid_span & op_valid

    # If none valid, return NaNs
    if not valid_mask.any():
        probs_empty = torch.full((N, p), float('nan'))
        H_empty     = torch.full((N,), float('nan'))
        sharp_empty = torch.full((N,), float('nan'))
        Hn_empty    = torch.full((N,), float('nan'))
        return probs_empty, H_empty, sharp_empty, Hn_empty, valid_mask.cpu()

    # ----- exact posterior P(t | x) per sample -----
    # w[i, t] = pi[t]*(p2[i]/(p-1)) for t!=x_i;    w[i, x_i] = pi[x_i]*(1-p2[i])
    pi = _prior_product_modp(p, device=device, dtype=dtype)  # [p]

    w = torch.zeros(N, p, dtype=dtype, device=device)
    # base fill for all t with p2/(p-1)
    w += (p2 / (p - 1)).view(-1, 1) * pi.view(1, -1)        # [N, p]
    # overwrite t = x with (1-p2)
    w.scatter_(1, product_obs.view(-1, 1), (pi[product_obs] * (1.0 - p2)).view(-1, 1))

    # zero-out invalid rows to avoid contaminating normalization
    invalid = ~valid_mask
    if invalid.any():
        w[invalid] = 0.0

    post_t = w / w.sum(dim=1, keepdim=True).clamp_min(eps)   # [N, p]

    # ----- pushforward via E3 = (t ± c) mod p -----
    y_idx = torch.arange(p, device=device).view(1, -1).expand(N, -1)  # [N, p]
    t_idx_plus  = (y_idx - c_val.view(-1, 1)) % p
    t_idx_minus = (y_idx + c_val.view(-1, 1)) % p

    probs_plus  = post_t.gather(1, t_idx_plus)   # [N, p]
    probs_minus = post_t.gather(1, t_idx_minus)  # [N, p]

    probs_y = torch.where(op_id.view(-1, 1) == PLUS_ID, probs_plus, probs_minus)  # [N, p]

    # normalize, then set invalid rows to NaN for clarity
    probs_y = probs_y / probs_y.sum(dim=1, keepdim=True).clamp_min(eps)
    probs_y[invalid] = float('nan')

    # ----- entropy-based sharpness -----
    P = probs_y.clamp_min(eps)
    H = -(P * P.log()).sum(dim=1)                                     # [N]
    sharp = torch.exp(H)                                              # [N]
    H_norm = H / torch.log(torch.tensor(float(p), device=device, dtype=dtype))

    # set invalid to NaN
    H[invalid] = float('nan')
    sharp[invalid] = float('nan')
    H_norm[invalid] = float('nan')

    return probs_y.cpu(), H.cpu(), sharp.cpu(), H_norm.cpu(), valid_mask.cpu()





def _prior_product_modp(p: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    Prior over true product t = (a*b) mod p with a,b ~ Unif{0..p-1}:
      pi[0]   = (2p-1)/p^2
      pi[t!=0]= (p-1)/p^2
    (Not explicitly needed in the E1+E2 posterior since we marginalize over (a,b),
     but kept for completeness.)
    """
    pi = torch.full((p,), (p - 1) / (p**2), dtype=dtype, device=device)
    pi[0] = (2 * p - 1) / (p**2)
    return pi



@torch.no_grad()
def exact_E3_entropy_sharpness_from_data_short_E1E2(
    data: torch.Tensor,         # [N, L] int64 tokens (short layout)
    p: int,                     # modulus p (digits 0..p-1)
    p1: torch.Tensor | float,   # scalar or [N] tensor: E1 noise prob (corrupt exactly one of {a,b})
    p2: torch.Tensor | float,   # scalar or [N] tensor: E2 noise prob (corrupt product only)
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Closed-form P(E3 | E1_obs, E2_obs) under your generator's noise model (SHORT layout).
    E1 noise: with prob p1, pick one of {a,b} uniformly and replace it with a random digit != true value.
    E2 noise: with prob p2, replace product t with a random digit != t.

    Steps (per row):
      1) Parse E1: a_obs, b_obs; Parse E2: x=product_obs, op, c (via first/second DED).
      2) Build E1 likelihood L1(a,b | a_obs,b_obs) under the "one-of-two is corrupted" model.
      3) Sum over all (a,b) with constraint (a*b mod p = t) to get unnormalized W0_t.
      4) Multiply by E2 likelihood L2(x | t), then normalize over t to get P(t | E1,E2).
      5) Pushforward via E3 = (t ± c) mod p to get P(E3 | E1,E2).
      6) Compute H, exp(H), H/log p.

    Returns (CPU tensors):
      probs_y : [N, p]  exact P(E3 | E1,E2) per row (NaN on invalid rows)
      H       : [N]     Shannon entropy over digits
      sharp   : [N]     entropy-based sharpness = exp(H)
      H_norm  : [N]     normalized entropy = H / log(p)
      valid_mask : [N]  rows where parsing is valid and op in {+, -}
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    device = data.device
    fdtype = torch.float64
    ldtype = torch.long

    # ---- Special token IDs (short layout vocabulary) ----
    PLUS_ID  = p
    MINUS_ID = p + 1
    DED_ID   = p + 5
    PAD_ID   = p + 7

    # ---- Broadcast p1, p2 to shape [N] ----
    def _to_vec(x):
        if isinstance(x, (float, int)):
            x = torch.tensor(x, device=device, dtype=fdtype)
        x = x.to(device=device, dtype=fdtype)
        return x.expand(N) if x.dim() == 0 else x
    p1 = _to_vec(p1)
    p2 = _to_vec(p2)

    # ---- Locate first non-PAD, first DED, second DED (vectorized) ----
    idx = torch.arange(L, device=device, dtype=ldtype).view(1, -1).expand(N, -1)  # [N, L]
    big = torch.full((N, L), L + 10, dtype=ldtype, device=device)

    nonpad_mask = (data != PAD_ID)
    start_idx = torch.where(nonpad_mask, idx, big).min(dim=1).values            # [N]

    ded_mask = (data == DED_ID)
    first_ded = torch.where(ded_mask, idx, big).min(dim=1).values               # [N]
    gt_first = idx > first_ded.view(-1, 1)
    second_ded = torch.where(ded_mask & gt_first, idx, big).min(dim=1).values   # [N]

    # Valid E2 span requires at least 3 tokens: product, op, c
    valid_span = (first_ded < big.max()) & (second_ded < big.max()) \
                 & (start_idx < first_ded) & (second_ded >= first_ded + 3)

    row = torch.arange(N, device=device, dtype=ldtype)

    # ---- Parse E1: a_obs at start+1, b_obs at start+3 (short layout) ----
    a_pos = (start_idx + 1).clamp_max(L - 1)
    b_pos = (start_idx + 3).clamp_max(L - 1)
    a_obs = torch.zeros(N, dtype=ldtype, device=device)
    b_obs = torch.zeros(N, dtype=ldtype, device=device)
    a_obs[valid_span] = data[row[valid_span], a_pos[valid_span]]
    b_obs[valid_span] = data[row[valid_span], b_pos[valid_span]]

    # Digits check for E1 tokens
    e1_digits_ok = (a_obs < p) & (b_obs < p)

    # ---- Parse E2: product, op, c from [first_ded+1 : first_ded+4) ----
    prod_pos = (first_ded + 1).clamp_max(L - 1)
    op_pos   = (first_ded + 2).clamp_max(L - 1)
    c_pos    = (first_ded + 3).clamp_max(L - 1)
    x_obs    = torch.zeros(N, dtype=ldtype, device=device)
    op_id    = torch.zeros(N, dtype=ldtype, device=device)
    c_val    = torch.zeros(N, dtype=ldtype, device=device)
    x_obs[valid_span] = data[row[valid_span], prod_pos[valid_span]]
    op_id[valid_span] = data[row[valid_span], op_pos[valid_span]]
    c_val[valid_span] = data[row[valid_span], c_pos[valid_span]]

    op_valid = (op_id == PLUS_ID) | (op_id == MINUS_ID)
    e2_digits_ok = (x_obs < p) & (c_val < p)

    valid_mask = valid_span & e1_digits_ok & e2_digits_ok & op_valid

    # If no valid rows, return NaNs
    if not valid_mask.any():
        return (
            torch.full((N, p), float('nan')),
            torch.full((N,), float('nan')),
            torch.full((N,), float('nan')),
            torch.full((N,), float('nan')),
            valid_mask.cpu(),
        )

    # ---- Precompute grids for all (a,b) and their product mod p ----
    a_vals = torch.arange(p, device=device, dtype=ldtype).view(p, 1).expand(p, p)  # [p, p]
    b_vals = torch.arange(p, device=device, dtype=ldtype).view(1, p).expand(p, p)  # [p, p]
    t_grid = (a_vals * b_vals) % p                                                 # [p, p]
    t_flat = t_grid.reshape(-1)                                                    # [p*p]

    # Constants used in E1 likelihood
    #   L1(a,b | a_obs,b_obs) =
    #     (1-p1)      if a=a_obs & b=b_obs
    #     (p1/2)/(p-1) if a!=a_obs & b=b_obs  OR  a=a_obs & b!=b_obs
    #     0           otherwise (can't corrupt both a and b)
    base_flip = (0.5) / max(p - 1, 1)  # (p1/2)/(p-1)

    # ---- Allocate outputs ----
    probs_y = torch.full((N, p), float('nan'), dtype=fdtype, device=device)
    H       = torch.full((N,), float('nan'), dtype=fdtype, device=device)
    sharp   = torch.full((N,), float('nan'), dtype=fdtype, device=device)
    H_norm  = torch.full((N,), float('nan'), dtype=fdtype, device=device)

    # ---- Process each valid row (O(p^2) per row; p≤~50 时非常快) ----
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    for i in valid_idx.tolist():
        ai = int(a_obs[i].item())
        bi = int(b_obs[i].item())
        xi = int(x_obs[i].item())
        ci = int(c_val[i].item())
        p1i = float(p1[i].item())
        p2i = float(p2[i].item())

        # E1 likelihood grid over (a,b)
        eqA = (a_vals == ai)
        eqB = (b_vals == bi)
        L1 = torch.zeros_like(t_grid, dtype=fdtype)
        L1[eqA & eqB] = (1.0 - p1i)
        L1[(~eqA) & eqB] = p1i * base_flip
        L1[eqA & (~eqB)] = p1i * base_flip
        # Note: both-different case remains 0

        # Sum L1 over all (a,b) that produce each t: W0[t] = sum_{a,b:ab=t} L1(a,b)
        W0 = torch.zeros(p, dtype=fdtype, device=device)
        W0.scatter_add_(0, t_flat, L1.reshape(-1))

        # E2 likelihood vector over t
        L2 = torch.full((p,), p2i / max(p - 1, 1), dtype=fdtype, device=device)
        L2[xi] = (1.0 - p2i)

        # Posterior over t given E1,E2 (unnormalized then normalized)
        post_t = (W0 * L2)
        post_t = post_t / post_t.sum().clamp_min(eps)

        # Pushforward via E3 = (t ± c) mod p
        if op_id[i].item() == PLUS_ID:
            # P_y[y] = P_t[(y - c) mod p]  -> roll by +c
            P_y = post_t.roll(shifts=ci)
        else:
            # minus: y = t - c -> P_y[y] = P_t[(y + c) mod p] -> roll by -c
            P_y = post_t.roll(shifts=-ci)

        P_y = (P_y / P_y.sum().clamp_min(eps)).to(fdtype)
        probs_y[i] = P_y

        # Entropy & sharpness
        P = P_y.clamp_min(eps)
        Hi = -(P * P.log()).sum()
        H[i] = Hi
        sharp[i] = torch.exp(Hi)
        H_norm[i] = Hi / torch.log(torch.tensor(float(p), dtype=fdtype, device=device))

    return probs_y.cpu(), H.cpu(), sharp.cpu(), H_norm.cpu(), valid_mask.cpu()







@torch.no_grad()
def measure_e3_agreement_with_e1_e2_modp(
    model,
    data: torch.Tensor,                 # [N, L] int64 tokens (SHORT layout: (a*b) +/- c)
    p: int,                             # modulus p (digits 0..p-1)
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    return_details: bool = False,
) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """
    SHORT-only version for (a * b) +/- c.

    Agreements computed against:
      - E1 simplified result: (a*b) op c  (all taken from the *observed* E1 tokens)
      - E2 simplified result: (product) op (c2)  (taken from the *observed* E2 triple between two DEDs)

    E3 prediction (model):
      - next-token style: read logits at position -3 (predicts token at -2), restrict to digits [0..p-1], argmax.
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # ---- Special token ids (mod-p vocabulary) ----
    PLUS_ID   = p
    MINUS_ID  = p + 1
    MUL_ID    = p + 4
    DED_ID    = p + 5
    PAD_ID    = p + 7

    # ---- Resolve device ----
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    was_training = model.training
    model.eval()

    # ===== 1) Model E3 predictions (argmax over digits using logits at -3) =====
    pred = torch.empty(N, dtype=torch.long)
    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = data[s:e].to(device, non_blocking=True)                 # [B, L]
        out = model(batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out      # [B, L, V]
        e3_logits = logits[:, -3, :][:, :p]                             # [B, p] digits only
        pred[s:e] = e3_logits.argmax(dim=-1).detach().cpu()

    # ===== 2) Parse E1/E2 (SHORT) and compute their simplified results =====
    e1_res = torch.full((N,), float('nan'))
    e2_res = torch.full((N,), float('nan'))
    valid_mask = torch.zeros(N, dtype=torch.bool)

    for i in range(N):
        seq = data[i]  # [L] on CPU

        # First non-PAD (start of expression)
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start = int(nonpad_idx[0].item())

        # SHORT layout pattern check:
        # [LP, a, MUL, b, RP, op, c, DED, product, op, c, DED, result, END]
        if start + 6 >= L:
            continue
        if int(seq[start + 2].item()) != MUL_ID:
            continue

        # ---- E1 simplified = (a*b) op c (mod p) ----
        a = int(seq[start + 1].item())
        b = int(seq[start + 3].item())
        op_tok = int(seq[start + 5].item())
        c1 = int(seq[start + 6].item())
        if not (0 <= a < p and 0 <= b < p and 0 <= c1 < p and op_tok in (PLUS_ID, MINUS_ID)):
            continue

        prod = (a * b) % p
        if op_tok == PLUS_ID:
            e1_val = (prod + c1) % p
        else:
            e1_val = (prod - c1) % p
        e1_res[i] = float(e1_val)

        # ---- E2 simplified from tokens between first and second DED: [product, op, c2] ----
        ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if ded_all.numel() < 2:
            continue
        first_ded_candidates = ded_all[ded_all >= start]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())
        second_ded_candidates = ded_all[ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())
        if second_ded < first_ded + 3:
            continue

        product_tok = int(seq[first_ded + 1].item())
        op2_tok     = int(seq[first_ded + 2].item())
        c2_tok      = int(seq[first_ded + 3].item())
        if not (0 <= product_tok < p and 0 <= c2_tok < p and op2_tok in (PLUS_ID, MINUS_ID)):
            continue

        if op2_tok == PLUS_ID:
            e2_val = (product_tok + c2_tok) % p
        else:
            e2_val = (product_tok - c2_tok) % p
        e2_res[i] = float(e2_val)

        valid_mask[i] = True

    # ===== 3) Agreement ratios on valid rows =====
    e1_valid = valid_mask & ~torch.isnan(e1_res)
    e2_valid = valid_mask & ~torch.isnan(e2_res)

    e1_pred_eq = torch.zeros(N, dtype=torch.bool)
    e2_pred_eq = torch.zeros(N, dtype=torch.bool)

    if e1_valid.any():
        e1_pred_eq[e1_valid] = (pred[e1_valid] == e1_res[e1_valid].to(torch.long))
    if e2_valid.any():
        e2_pred_eq[e2_valid] = (pred[e2_valid] == e2_res[e2_valid].to(torch.long))

    e1_agree_rate = float(e1_pred_eq[e1_valid].float().mean().item()) if e1_valid.any() else float('nan')
    e2_agree_rate = float(e2_pred_eq[e2_valid].float().mean().item()) if e2_valid.any() else float('nan')

    extras: Dict[str, torch.Tensor] = {}
    if return_details:
        extras = {
            "pred": pred,                     # [N]
            "e1_result": e1_res,              # [N], float (NaN where invalid)
            "e2_result": e2_res,              # [N], float (NaN where invalid)
            "e1_eq": e1_pred_eq,              # [N], bool
            "e2_eq": e2_pred_eq,              # [N], bool
            "valid_mask": valid_mask,         # [N], bool
        }

    if was_training:
        model.train()

    return e1_agree_rate, e2_agree_rate, extras






@torch.no_grad()
def compute_e3_entropy_sharpness_modp_short(
    model,
    data: torch.Tensor,               # [N, L] int64 tokens
    p: int,                           # modulus p (digits are 0..p-1; specials start at p)
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    e3_logits_pos: int = -3,          # read logits at -3 (to predict the E3 token at -2)
) -> Tuple[torch.Tensor, float]:
    """
    Entropy-based sharpness at E3 (short layout), ignoring ground truth:
      sharpness = exp(H(P_digits)), where P_digits is the softmax over digit classes [0..p-1]
      taken from the E3 logits position (default -3).

    Returns:
      per_sample_sharp: [N] tensor, exp(H) per sample (CPU)
      sharp_mean:       float, mean exp(H) across samples
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    was_training = model.training
    model.eval()

    per_sample_sharp = torch.empty(N, dtype=torch.float32)

    eps = 1e-12
    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = data[s:e].to(device, non_blocking=True)      # [B, L]

        # Forward pass
        out = model(batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out   # [B, L, V]

        # Take E3 logits at -3 and restrict to digit classes [0..p-1]
        e3_logits = logits[:, e3_logits_pos, :][:, :p]       # [B, p]

        # P_digits: distribution over digits only
        P = F.softmax(e3_logits, dim=-1)                     # [B, p]

        # Shannon entropy H(P) over digits and entropy-based sharpness exp(H)
        H = -(P * (P.clamp_min(eps).log())).sum(dim=-1)      # [B]
        sharp = torch.exp(H).float().cpu()                   # [B]

        per_sample_sharp[s:e] = sharp

    sharp_mean = float(per_sample_sharp.mean().item())

    if was_training:
        model.train()

    return per_sample_sharp, sharp_mean







###### (a*b)+/-c data
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

@torch.no_grad()
def autoreg_generate_from_firstded_to_stop_1(
    model,
    data: torch.Tensor,                   # [N, L] int64 tokens
    device: Optional[torch.device] = None,
    decode_strategy: str = "sample",      # "greedy" or "sample"
    temperature: float = 1.0,             # used when decode_strategy == "sample"
    p: int = 11
) -> torch.Tensor:
    
    pad_token_id = p + 7
    stop_token_id = p + 6
    ded_token_id = p + 5
    
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    if pad_token_id is None or stop_token_id is None or ded_token_id is None:
        raise ValueError("pad_token_id, stop_token_id, and ded_token_id must be provided.")

    # resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    # move to device; keep a working copy
    x = data.to(device, non_blocking=True).clone()
    was_training = model.training
    model.eval()

    for i in range(N):
        seq = x[i].clone()

        # find first non-PAD index
        nonpad = (seq != pad_token_id).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            x[i] = seq
            continue
        start_idx = int(nonpad[0].item())

        # find all DEDUCT positions
        ded_pos = (seq == ded_token_id).nonzero(as_tuple=False).squeeze(-1)
        if ded_pos.numel() == 0:
            x[i] = seq
            continue

        # first DEDUCT at/after start_idx
        cand = ded_pos[ded_pos >= start_idx]
        if cand.numel() == 0:
            x[i] = seq
            continue
        first_ded = int(cand[0].item())

        gen_seq = seq.clone()
        pos = first_ded + 1

        # autoregressive loop
        while pos < L:
            # mask future positions to PAD to avoid leakage
            inp = gen_seq.clone()
            if pos + 1 < L:
                inp[pos + 1:] = pad_token_id

            # forward pass
            logits, _ = model(inp.unsqueeze(0))  # [1, L, V]

            # IMPORTANT: assume logits[t] predicts token at position t+1
            # so to predict token at index `pos`, read logits at index (pos-1)
            if pos - 1 < 0:
                # degenerate safety: if DEDUCT at position 0 (should not happen)
                next_logits = logits[0, pos, :]
            else:
                next_logits = logits[0, pos - 1, :]

            # decode
            if decode_strategy == "greedy":
                nxt = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            gen_seq[pos] = nxt

            # stopping condition
            if nxt == stop_token_id:
                if pos + 1 < L:
                    gen_seq[pos + 1:] = pad_token_id
                break

            pos += 1

        x[i] = gen_seq

    if was_training:
        model.train()

    return x


def eval_expr_modp_1(tokens: torch.Tensor, p: int = 11) -> Optional[int]:
    """
    Evaluate an infix arithmetic expression under mod p.

    Assumes vocabulary layout consistent with build_modular_vocabulary(p):
      digits: 0 .. p-1
      PLUS : p
      MINUS: p+1
      LP   : p+2
      RP   : p+3
      MUL  : p+4
      DEDUCT: p+5   (not part of expression segments we pass in)
      END  : p+6    (not part of expression segments we pass in)
      PAD  : p+7

    Any token that is not digit / + / - / * / parentheses
    makes the given segment invalid for evaluation.
    """

    PLUS_ID  = p
    MINUS_ID = p + 1
    LP_ID    = p + 2
    RP_ID    = p + 3
    MUL_ID   = p + 4

    if tokens.numel() == 0:
        return None

    xs = tokens.tolist()
    prec = {PLUS_ID: 1, MINUS_ID: 1, MUL_ID: 2}

    def is_digit(t): return 0 <= t <= p - 1
    def is_op(t):    return t in (PLUS_ID, MINUS_ID, MUL_ID)

    # infix -> RPN (shunting-yard)
    out, st = [], []
    for t in xs:
        if is_digit(t):
            out.append(t)
        elif t == LP_ID:
            st.append(t)
        elif t == RP_ID:
            while st and st[-1] != LP_ID:
                out.append(st.pop())
            if not st or st[-1] != LP_ID:
                return None
            st.pop()  # pop '('
        elif is_op(t):
            while st and st[-1] in prec and prec[st[-1]] >= prec[t]:
                out.append(st.pop())
            st.append(t)
        else:
            # any other token is invalid in expression context
            return None

    while st:
        if st[-1] in (LP_ID, RP_ID):
            return None
        out.append(st.pop())

    # evaluate RPN under mod p
    stk = []
    for t in out:
        if is_digit(t):
            stk.append(t % p)
        else:
            if len(stk) < 2:
                return None
            b = stk.pop()
            a = stk.pop()
            if   t == PLUS_ID:  stk.append((a + b) % p)
            elif t == MINUS_ID: stk.append((a - b) % p)
            elif t == MUL_ID:   stk.append((a * b) % p)
            else:               return None

    if len(stk) != 1:
        return None
    return int(stk[0] % p)



def proportion_E1_eq_r_but_E2_neq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:

    # IDs implied by build_modular_vocabulary(p)
    PLUS_ID    = p
    MINUS_ID   = p + 1
    LP_ID      = p + 2
    RP_ID      = p + 3
    MUL_ID     = p + 4
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        # first non-PAD index
        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        # find all DEDUCT positions
        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        # first DEDUCT at/after start, then the next one
        first_deds = deds[deds >= start]
        if first_deds.numel() == 0:
            continue
        first_ded = int(first_deds[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        # r should be the token right after the second DEDUCT
        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            # r must be a digit in [0, p-1]
            continue

        # E1 tokens: [start, first_ded)
        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        # E2 tokens: (first_ded, second_ded)
        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        # evaluate
        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        # valid sample
        denom_mask[i] = True
        count_den += 1

        # numerator condition
        if (e1_val == r_tok) and (e2_val != r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask





def proportion_E1_neq_r_but_E2_eq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:
    
    """
    ratio = P( E1 != r  AND  E2 == r | sample is well-formed )
    Returns (ratio, denom_mask).
    """
    
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        # first non-PAD index
        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        # find DEDUCT positions
        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            continue

        # E1 = [start, first_ded)
        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        # E2 = (first_ded, second_ded)
        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        # evaluate expressions mod p
        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        denom_mask[i] = True
        count_den += 1

        if (e1_val != r_tok) and (e2_val == r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask


def proportion_E1_eq_r_and_E2_eq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:
    """
    ratio = P( E1 == r  AND  E2 == r | sample is well-formed )
    Returns (ratio, denom_mask).
    """
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            continue

        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        denom_mask[i] = True
        count_den += 1

        if (e1_val == r_tok) and (e2_val == r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask


def proportion_E1_neq_r_and_E2_neq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:
    """
    ratio = P( E1 != r  AND  E2 != r | sample is well-formed )
    Returns (ratio, denom_mask).
    """
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            continue

        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        denom_mask[i] = True
        count_den += 1

        if (e1_val != r_tok) and (e2_val != r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask





#partial replacement conditioning on consistency with solution
@torch.no_grad()
def measure_e2_addup_intervention_ce_modp_short(
    model,
    data: torch.Tensor,                       
    p: int,                                   
    num_variants: int = 50,                   # number of variants per eligible sample
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    seed: Optional[int] = None, 
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    
    # --- Helpers ---
    def _is_digit(tok: int) -> bool:
        return 0 <= tok < p

    def _modp(x: int) -> int:
        return int(x % p)


    def _parse_E1(seq: torch.Tensor, b0: int, b1: int):
        # find MUL position and surrounding digits
        mul_pos = None
        for i in range(b0, b1):
            if int(seq[i].item()) == MUL_ID:
                mul_pos = i
                break
        if mul_pos is None:
            return None

        # digit immediately before mul
        a_pos = None
        for i in range(mul_pos - 1, b0 - 1, -1):
            if _is_digit(int(seq[i].item())):
                a_pos = i
                break
        # digit immediately after mul
        b_pos = None
        for i in range(mul_pos + 1, b1):
            if _is_digit(int(seq[i].item())):
                b_pos = i
                break
        if a_pos is None or b_pos is None:
            return None

        # find PLUS/MINUS after mul
        op_pos = None
        for i in range(mul_pos + 1, b1):
            tok = int(seq[i].item())
            if tok == PLUS_ID or tok == MINUS_ID:
                op_pos = i
                break
        if op_pos is None:
            return None

        # digit after op
        c_pos = None
        for i in range(op_pos + 1, b1):
            if _is_digit(int(seq[i].item())):
                c_pos = i
                break
        if c_pos is None:
            return None

        a = int(seq[a_pos].item())
        b = int(seq[b_pos].item())
        op = int(seq[op_pos].item())
        c = int(seq[c_pos].item())
        return (a, b, op, c)


    def _parse_E2(seq: torch.Tensor, b0: int, b1: int):
        op_pos = None
        for i in range(b0, b1):
            tok = int(seq[i].item())
            if tok == PLUS_ID or tok == MINUS_ID:
                op_pos = i
                break
        if op_pos is None:
            return None
        # digit before op
        d_pos = None
        for i in range(op_pos - 1, b0 - 1, -1):
            if _is_digit(int(seq[i].item())):
                d_pos = i
                break
        # digit after op
        c_pos = None
        for i in range(op_pos + 1, b1):
            if _is_digit(int(seq[i].item())):
                c_pos = i
                break
        if d_pos is None or c_pos is None:
            return None

        d = int(seq[d_pos].item())
        op = int(seq[op_pos].item())
        c = int(seq[c_pos].item())
        return (d_pos, op_pos, c_pos, d, op, c)

    # compute E1 and E2 results mod p
    def _eval_E1_modp(a: int, b: int, op: int, c: int) -> int:
        ab = _modp(a * b)
        if op == PLUS_ID:
            return _modp(ab + c)
        elif op == MINUS_ID:
            return _modp(ab - c)
        else:
            return None

    def _eval_E2_modp(d: int, op: int, c: int) -> int:
        if op == PLUS_ID:
            return _modp(d + c)
        elif op == MINUS_ID:
            return _modp(d - c)
        else:
            return None

    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Restrict to digit classes [0..p-1]
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]; keep aligned with your previous code

   
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    was_training = model.training
    model.eval()

    
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]

    
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    for i in range(N):
        seq = data[i]  # CPU 1D

        # First non-PAD
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # Find two '->' as DED_ID
        ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if ded_all.numel() < 2:
            continue
        first_ded_candidates = ded_all[ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = ded_all[ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # Segment ranges
        e1_b0, e1_b1 = start_idx, first_ded
        e2_b0, e2_b1 = first_ded + 1, second_ded

        # Parse E1 and E2
        e1_parsed = _parse_E1(seq, e1_b0, e1_b1)
        e2_parsed = _parse_E2(seq, e2_b0, e2_b1)
        if e1_parsed is None or e2_parsed is None:
            continue

        a, b, op1, c1 = e1_parsed
        d_pos, op2_pos, c2_pos, d, op2, c2 = e2_parsed

        # Require E1 result == E2 result (mod p) BEFORE intervention
        e1_val = _eval_E1_modp(a, b, op1, c1)
        e2_val = _eval_E2_modp(d, op2, c2)
        if e1_val is None or e2_val is None:
            continue
        if e1_val != e2_val:
            # skip this sample: only intervene when E1 and E2 agree
            continue

        # Build num_variants interventions
        built = 0
        tries = 0
        max_tries = num_variants * 6  # avoid infinite loops if bounds tight
        while built < num_variants and tries < max_tries:
            tries += 1

            new_seq = seq.clone()
            new_d, new_c = d, c2

            if op2 == PLUS_ID:
                # reverse shift:
                # Option A: d += s, c -= s   with 1 <= s <= min(p-1-d, c)
                # Option B: d -= s, c += s   with 1 <= s <= min(d, p-1-c)
                import random
                if random.random() < 0.5:
                    s_max = min((p - 1 - new_d), new_c)
                    if s_max >= 1:
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d + s
                        new_c = new_c - s
                    else:
                        # try the other direction
                        s_max = min(new_d, (p - 1 - new_c))
                        if s_max < 1:
                            continue
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d - s
                        new_c = new_c + s
                else:
                    s_max = min(new_d, (p - 1 - new_c))
                    if s_max >= 1:
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d - s
                        new_c = new_c + s
                    else:
                        # try the other direction
                        s_max = min((p - 1 - new_d), new_c)
                        if s_max < 1:
                            continue
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d + s
                        new_c = new_c - s

            elif op2 == MINUS_ID:
                # same-direction shift:
                # Option A: d += s, c += s   with 1 <= s <= min(p-1-d, p-1-c)
                # Option B: d -= s, c -= s   with 1 <= s <= min(d, c)
                import random
                if random.random() < 0.5:
                    s_max = min((p - 1 - new_d), (p - 1 - new_c))
                    if s_max >= 1:
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d + s
                        new_c = new_c + s
                    else:
                        s_max = min(new_d, new_c)
                        if s_max < 1:
                            continue
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d - s
                        new_c = new_c - s
                else:
                    s_max = min(new_d, new_c)
                    if s_max >= 1:
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d - s
                        new_c = new_c - s
                    else:
                        s_max = min((p - 1 - new_d), (p - 1 - new_c))
                        if s_max < 1:
                            continue
                        s = torch.randint(1, s_max + 1, (1,)).item()
                        new_d = new_d + s
                        new_c = new_c + s
            else:
                # not plus/minus; skip
                continue

            # Bounds check (redundant but safe)
            if not (0 <= new_d < p and 0 <= new_c < p):
                continue

            # Keep E2 value unchanged mod p (should hold by construction)
            new_val = _eval_E2_modp(new_d, op2, new_c)
            if new_val != e2_val:
                # If something went off, skip
                continue

            # Apply to sequence
            new_seq[d_pos] = int(new_d)
            new_seq[c2_pos] = int(new_c)

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)
            built += 1

    # No interventions possible
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    # Stack and score
    modified_tensor = torch.stack(modified_seqs, dim=0)                                   # [M, L] CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)      # [M] on device
    M = modified_tensor.shape[0]

    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]

            # Cross-entropy CE(P, Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # Invariance: argmax preserved?
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()











#adversarial intervention
@torch.no_grad()
def measure_e2_max_intervention_kl_modp_short(
    model,
    data: torch.Tensor,                       
    p: int,                                   
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:

    def _is_digit(tok: int) -> bool:
        return 0 <= tok < p

    def _parse_E2(seq: torch.Tensor, b0: int, b1: int):
        op_pos = None
        for i in range(b0, b1):
            tok = int(seq[i].item())
            if tok == PLUS_ID or tok == MINUS_ID:
                op_pos = i
                break
        if op_pos is None:
            return None
        
        d_pos = None
        for i in range(op_pos - 1, b0 - 1, -1):
            if _is_digit(int(seq[i].item())):
                d_pos = i
                break
        
        c_pos = None
        for i in range(op_pos + 1, b1):
            if _is_digit(int(seq[i].item())):
                c_pos = i
                break
        if d_pos is None or c_pos is None:
            return None
        d = int(seq[d_pos].item())
        op = int(seq[op_pos].item())
        c = int(seq[c_pos].item())
        return (d_pos, op_pos, c_pos, d, op, c)


    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        return model_outputs_logits[:, -3, :][:, :p]  


    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    was_training = model.training
    model.eval()

    
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]
    logP = torch.log(P + 1e-12)

    
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    for i in range(N):
        seq = data[i]  # CPU 1D

       
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        
        ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if ded_all.numel() < 2:
            continue
        first_ded_candidates = ded_all[ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = ded_all[ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        
        e2_b0, e2_b1 = first_ded + 1, second_ded
        parsed = _parse_E2(seq, e2_b0, e2_b1)
        if parsed is None:
            continue
        d_pos, _, _, d_orig, _, _ = parsed

        
        for new_d in range(p):
            if new_d == d_orig:
                continue
            new_seq = seq.clone()
            new_seq[d_pos] = int(new_d)
            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN


    modified_tensor = torch.stack(modified_seqs, dim=0)                                   # [M, L] CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)      # [M]
    M = modified_tensor.shape[0]

    # best trackers
    neg_inf = float("-inf")
    best_kl_per_sample = torch.full((N,), neg_inf, device=device, dtype=torch.float32)
    best_inv_per_sample = torch.zeros(N, device=device, dtype=torch.float32)  # invariance of the best
    has_best = torch.zeros(N, device=device, dtype=torch.bool)

    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]
            logP_sel = logP[ori_idx]                                     # [B, p]

            # KL(P || Q) = sum P * (logP - logQ)
            kl_vec = (P_sel * (logP_sel - logQ)).sum(dim=-1)             # [B]

            # invariance for this candidate
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            # update best per sample
            for j in range(kl_vec.shape[0]):
                idx = int(ori_idx[j].item())
                val = float(kl_vec[j].item())
                if val > float(best_kl_per_sample[idx].item()):
                    best_kl_per_sample[idx] = val
                    best_inv_per_sample[idx] = inv_vec[j]
                    has_best[idx] = True

    # reduce to selected samples
    selected_mask = has_best
    num_sel = int(selected_mask.sum().item())
    if num_sel == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    avg_kl = float(best_kl_per_sample[selected_mask].mean().item())
    inv_mean = float(best_inv_per_sample[selected_mask].mean().item())

    # fill non-selected with 0 for per-sample outputs
    per_sample_kl = torch.zeros(N, dtype=torch.float32)
    per_sample_inv = torch.zeros(N, dtype=torch.float32)
    per_sample_kl[selected_mask.cpu()] = best_kl_per_sample[selected_mask].detach().cpu()
    per_sample_inv[selected_mask.cpu()] = best_inv_per_sample[selected_mask].detach().cpu()

    if was_training:
        model.train()

    return avg_kl, per_sample_kl, inv_mean, per_sample_inv








#

def _flatten_params(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])





def _zeros_like_param_flat(p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p, dtype=p.dtype, device=p.device).reshape(-1)




def _hvp_from_loss(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    v: torch.Tensor,
) -> torch.Tensor:
    
    g1 = grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    
    g1_flat = torch.cat([
        (g.reshape(-1) if g is not None else _zeros_like_param_flat(p))
        for g, p in zip(g1, params)
    ])
    
    gv = (g1_flat * v).sum()
    
    g2 = grad(gv, params, retain_graph=False, allow_unused=True)
    hv = torch.cat([
        (g.reshape(-1) if g is not None else _zeros_like_param_flat(p))
        for g, p in zip(g2, params)
    ])
    
    return hv







def estimate_hessian_top_eig_with_get_loss(
    model: nn.Module,
    criterion: nn.Module,
    src: torch.Tensor,               
    mask: Optional[torch.Tensor] = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    return_eigvec: bool = False,
) -> Tuple[float, Optional[torch.Tensor]]:
  
    torch.set_grad_enabled(True)

    was_training = model.training
    model.eval() 

    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0, "No trainable parameters were found."

    flat0 = _flatten_params(params)
    dim = flat0.numel()
    device = flat0.device
    dtype = flat0.dtype

    if seed is not None:
        torch.manual_seed(seed)

    v = torch.randn(dim, device=device, dtype=dtype)
    v /= (v.norm() + 1e-12)

    last_rayleigh = None

    for it in range(max_iter):
        loss = get_loss(model, criterion, src, mask)
        assert loss.dim() == 0, "get_loss must return a scalar loss."

        hv = _hvp_from_loss(loss, params, v)
        hv_norm = hv.norm()

        if hv_norm.item() == 0.0:
            lambda_est = 0.0
            break

        lambda_est = torch.dot(v, hv).item()

        v = (hv / hv_norm).detach()

        if last_rayleigh is not None:
            rel = abs(lambda_est - last_rayleigh) / (abs(last_rayleigh) + 1e-12)
            if rel < tol and it >= 3:
                break
                
        last_rayleigh = lambda_est

    loss = get_loss(model, criterion, src, mask)
    hv = _hvp_from_loss(loss, params, v)
    lambda_max = torch.dot(v, hv).item()

    if was_training:
        model.train()

    return lambda_max







@torch.no_grad()
def second_layer_attn_ffn_no_residual(
    model,
    src: torch.Tensor,
    *,
    use_training_mode: bool = False,
    return_attn: bool = False
):

    prev_training = model.training
    model.train(mode=use_training_mode)

    device = getattr(model, "device", src.device)
    src = src.to(device).long()

    if model.pos in ["relative", "rotary"]:
        out = model.embed(src)
    else:
        out = model.pos_embed(model.embed(src))

    B, T, _ = out.shape

    # masks (mirror TFModel.forward)
    
    pad_mask = (src != model.pad_id)
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask.unsqueeze(1).unsqueeze(2)

    # first layer fully
    if len(model.h_1) == 0:
        raise RuntimeError("one layer only")
    out, _ = model.h_1[0](out, mask)

    b = model.h_1[1] if len(model.h_1) >= 2 else model.h_2

    attn_in = b.ln_1(out) if b.norm else out
    attn_out, attn_info = b.mha(attn_in, attn_in, attn_in, mask, output_attn=True)
    attn_out = b.dropout1(attn_out) if b.drop is not None else attn_out
    x_attn = out + attn_out if b.residual else out  # <-- residual here

    pre_ffn = b.ln_2(x_attn) if b.norm else x_attn
    ffn_out = b.feed_forward(pre_ffn)
    ffn_out = b.dropout2(ffn_out) if b.drop is not None else ffn_out

    
    model.train(mode=prev_training)

    
    if return_attn:
        return ffn_out, attn_info  #[B,T,d_model], (attn_probs, QK_vals)
    
    
    
    return ffn_out





import torch

@torch.no_grad()
def third_layer_attn_ffn_no_residual(
    model,
    src: torch.Tensor,
    *,
    use_training_mode: bool = False,
    return_attn: bool = False
):
    # --- keep/restore mode ---
    prev_training = model.training
    model.train(mode=use_training_mode)

    # --- device & dtype ---
    device = getattr(model, "device", src.device)
    src = src.to(device).long()

    # --- embeddings (mirror TFModel.forward) ---
    if getattr(model, "pos", None) in ["relative", "rotary"]:
        out = model.embed(src)
    else:
        out = model.pos_embed(model.embed(src))

    B, T, _ = out.shape

    # --- masks (mirror TFModel.forward) ---
    pad_mask = (src != model.pad_id)
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask.unsqueeze(1).unsqueeze(2)

    # --- run layer 1 & 2 fully from h_1 ---
    if not hasattr(model, "h_1") or len(model.h_1) < 2:
        model.train(mode=prev_training)
        raise RuntimeError("h_1 must contain two layers to reach layer 3")
    out, _ = model.h_1[0](out, mask)
    out, _ = model.h_1[1](out, mask)

    # --- select third layer from model.h_2 ---
    if not hasattr(model, "h_2"):
        model.train(mode=prev_training)
        raise RuntimeError("Third layer not found (expect model.h_2)")
    b = model.h_2

    # --- manual MHA + FFN for layer 3 ---
    # norm -> mha -> dropout1 -> (residual) -> norm -> ffn -> dropout2
    attn_in = b.ln_1(out) if getattr(b, "norm", False) else out
    attn_out, attn_info = b.mha(attn_in, attn_in, attn_in, mask, output_attn=True)
    attn_out = b.dropout1(attn_out) if getattr(b, "drop", None) is not None else attn_out
    x_attn = out + attn_out if getattr(b, "residual", True) else out

    pre_ffn = b.ln_2(x_attn) if getattr(b, "norm", False) else x_attn
    ffn_out = b.feed_forward(pre_ffn)
    ffn_out = b.dropout2(ffn_out) if getattr(b, "drop", None) is not None else ffn_out


    
    # --- restore mode ---
    model.train(mode=prev_training)

    return ffn_out







#negative sample generator
@torch.no_grad()
def shuffle_e2_first_digit_modp_short(
    data: torch.Tensor,                       # [N, L] int64 tokens (short expression sequences)
    p: int,                                   # modulus p (digits are 0..p-1)
    digits_for_replace: Optional[Sequence[int]] = None,  # allowed digits for replacement
    seed: Optional[int] = None,
) -> torch.Tensor:

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Work on a clone so we don't modify the input in-place
    out = data.clone()  # same device as data

    # Special token IDs for mod p (we actually only need DED and PAD)
    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    if seed is not None:
        torch.manual_seed(seed)

    # Allowed digits for replacement
    if digits_for_replace is None:
        # Default: avoid 0 and 5 when p > 5, else avoid 0 only (same as your original logic)
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = out[i]  # [L], same device as data

        # 1) Find first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # 2) Locate the two DED markers delimiting E2 in short layout
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() < 2:
            continue

        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = idx_ded_all[idx_ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # E2 must contain at least one token
        if second_ded <= first_ded + 1:
            continue

        e2_slice = seq[first_ded + 1 : second_ded]

        # 3) Identify the first digit token in E2 (value < p)
        rel_digit_positions = (e2_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digit_positions.numel() == 0:
            # No digit in E2; skip this sample
            continue
        first_rel_digit = int(rel_digit_positions[0].item())
        product_abs_pos = first_ded + 1 + first_rel_digit

        orig_product = int(seq[product_abs_pos].item())

        # 4) Build candidate set (exclude the original digit when possible)
        if orig_product in allowed_digits and len(allowed_digits) > 1:
            candidates = [d for d in allowed_digits if d != orig_product]
        else:
            candidates = allowed_digits

        if not candidates:
            continue

        # 5) Sample exactly one new digit and replace
        new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
        seq[product_abs_pos] = new_digit

    return out








@torch.no_grad()
def get_last_layer_QK(
    model,
    src: torch.LongTensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:

    if device is None:
        device = getattr(model, "device", src.device)

    model.eval()
    src = src.to(device)

    # ---- 1) do exactly the same embedding + mask as TFModel.forward ----
    if model.pos in ["relative", "rotary"]:
        x = model.embed(src)
    else:
        x = model.pos_embed(model.embed(src))  # [B, T, d_model]

    batch_size, seq_len, _ = x.size()

    pad_mask = (src != model.pad_id)
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    )
    # shape: [B, 1, 1, T] & [1, 1, T, T] -> [B, 1, T, T]
    mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask.unsqueeze(1).unsqueeze(2)

    # ---- 2) run all blocks in h_1 to get the input to the last layer ----
    out = x
    for block in model.h_1:
        out, _ = block(out, mask)   # we ignore attn_probs here

    # ---- 3) on the last block h_2, compute Q, K and get QK_vals ----
    last_block = model.h_2

    # input to MHA in TFBlock.forward: ln_1(x) if norm else x
    x_mha_in = last_block.ln_1(out) if last_block.norm else out  # [B, T, d_model]

    mha = last_block.mha
    # same as MultiHeadAttention.forward but we intercept before softmax
    Q = mha.split_heads(mha.W_q(x_mha_in))  # [B, num_heads, T, d_k]
    K = mha.split_heads(mha.W_k(x_mha_in))  # [B, num_heads, T, d_k]
    V = mha.split_heads(mha.W_v(x_mha_in))  # [B, num_heads, T, d_k] (not really needed but required by API)

    # scaled_dot_product_attention returns (output, (attn_probs, QK_vals))
    attn_output, (attn_probs, QK_vals) = mha.scaled_dot_product_attention(Q, K, V, mask)

    
    return QK_vals.mean(dim=0)

















































































def select_E1_a_in_firstN(
    seqs: torch.Tensor,
    a_values: Sequence[int],
    max_n: int,
    p: int = 11,
) -> torch.Tensor:
    """
    Select the first max_n sequences whose E1 segment has
    first digit a in a_values.

    Assumes sequence structure compatible with:
        PAD*  E1  DEDUCT  E2  DEDUCT  r  END  PAD*

    Vocabulary (build_modular_vocabulary):
      digits: 0 .. p-1
      PLUS : p
      MINUS: p+1
      LP   : p+2
      RP   : p+3
      MUL  : p+4
      DEDUCT: p+5
      END  : p+6
      PAD  : p+7
    """

    # IDs
    DEDUCT_ID = p + 5
    PAD_ID    = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)
    N, L = seqs.shape

    allowed_a = set(int(v) for v in a_values)
    selected_indices = []

    for i in range(N):
        if len(selected_indices) >= max_n:
            break

        row = seqs[i]

        # 1) Find first non-PAD index -> start of E1
        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        # 2) Find first DEDUCT at/after start -> right boundary of E1
        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 1:
            continue
        first_deds = deds[deds >= start]
        if first_deds.numel() == 0:
            continue
        first_ded = int(first_deds[0].item())
        if first_ded <= start:
            continue

        # 3) E1 tokens = [start, first_ded)
        e1_tokens = row[start:first_ded]

        # 4) First digit in E1 = a
        a_tok = None
        for t in e1_tokens.tolist():
            if 0 <= t <= p - 1:  # digit
                a_tok = t
                break
        if a_tok is None:
            continue

        # 5) Check membership
        if a_tok in allowed_a:
            selected_indices.append(i)

    if len(selected_indices) == 0:
        # Return empty tensor with same feature dimension
        return seqs.new_empty((0, L))

    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=seqs.device)
    return seqs[selected_indices]





@torch.no_grad()
def autoreg_generate_from_secondded_to_stop_1(
    model,
    data: torch.Tensor,                   # [N, L] int64 tokens
    device: Optional[torch.device] = None,
    decode_strategy: str = "greedy",      # "greedy" or "sample"
    temperature: float = 1.0,             # used when decode_strategy == "sample"
    p: int = 11
) -> torch.Tensor:
    """
    Autoregressively generate tokens starting from the SECOND DEDUCT token.

    For your dataset:
      tokens: [E1 tokens..., DEDUCT, E2 tokens..., DEDUCT, result, END, PAD...]
    We:
      - Find first non-PAD
      - Find all DEDUCT at/after that index
      - If there are at least two, take the SECOND one
      - Start generation from position (second_ded + 1)
      - Stop when END is generated, then pad the rest.

    Pass the correct IDs from ModularVocabulary, e.g. for p=11:
      PLUS=11, MINUS=12, LP=13, RP=14, MUL=15,
      DEDUCT=16, END=17, PAD=18
    """

    pad_token_id = p + 7
    stop_token_id = p + 6
    ded_token_id = p + 5

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), \
        "data must be [N, L] int64"
    N, L = data.shape

    # resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    # move to device; keep a working copy
    x = data.to(device, non_blocking=True).clone()
    was_training = model.training
    model.eval()

    for i in range(N):
        seq = x[i].clone()

        # find first non-PAD index
        nonpad = (seq != pad_token_id).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            # all PAD, nothing to do
            x[i] = seq
            continue
        start_idx = int(nonpad[0].item())

        # find all DEDUCT positions
        ded_pos = (seq == ded_token_id).nonzero(as_tuple=False).squeeze(-1)
        if ded_pos.numel() == 0:
            # no DED at all
            x[i] = seq
            continue

        # keep only DEDUCT positions at/after start_idx
        cand = ded_pos[ded_pos >= start_idx]
        if cand.numel() < 2:
            # fewer than 2 DED tokens -> do nothing
            x[i] = seq
            continue

        # SECOND DEDUCT at/after start_idx
        second_ded = int(cand[1].item())

        gen_seq = seq.clone()
        pos = second_ded + 1

        # autoregressive loop
        while pos < L:
            # mask future positions to PAD to avoid leakage
            inp = gen_seq.clone()
            if pos + 1 < L:
                inp[pos + 1:] = pad_token_id

            # forward pass
            logits, _ = model(inp.unsqueeze(0))  # [1, L, V]

            # assume logits[t] predicts token at position t+1
            # to predict token at index `pos`, read logits at index (pos-1)
            if pos - 1 < 0:
                # degenerate safety
                next_logits = logits[0, pos, :]
            else:
                next_logits = logits[0, pos - 1, :]

            # decode
            if decode_strategy == "greedy":
                nxt = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            gen_seq[pos] = nxt

            # stopping condition
            if nxt == stop_token_id:
                if pos + 1 < L:
                    gen_seq[pos + 1:] = pad_token_id
                break

            pos += 1

        x[i] = gen_seq

    if was_training:
        model.train()

    return x





def proportion_E1_eq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:
    """
    ratio = P( E1 != r  AND  E2 != r | sample is well-formed )
    Returns (ratio, denom_mask).
    """
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            continue

        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        if e1_val is None:
            continue

        denom_mask[i] = True
        count_den += 1

        if (e1_val == r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask

def proportion_E2_eq_r_1(
    seqs: torch.Tensor,
    p: int = 11,
) -> Tuple[float, torch.Tensor]:
    """
    ratio = P( E1 != r  AND  E2 != r | sample is well-formed )
    Returns (ratio, denom_mask).
    """
    DEDUCT_ID  = p + 5
    END_ID     = p + 6
    PAD_ID     = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)

    N, L = seqs.shape
    denom_mask = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    count_num = 0
    count_den = 0

    for i in range(N):
        row = seqs[i]

        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        r_pos = second_ded + 1
        if r_pos >= L:
            continue
        r_tok = int(row[r_pos].item())
        if not (0 <= r_tok <= p - 1):
            continue

        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e2_val is None:
            continue

        denom_mask[i] = True
        count_den += 1

        if (e2_val == r_tok):
            count_num += 1

    ratio = (count_num / count_den) if count_den > 0 else 0.0
    return ratio, denom_mask







def select_E1_a_and_E1_not_eq_E2_og(
    seqs: torch.Tensor,
    a_values: Sequence[int],
    p: int = 11,
    max_n: int = 1000,  
) -> torch.Tensor:
    """
    Return up to max_n sequences where:
      - The first digit in E1 ∈ a_values
      - eval(E1) ≠ eval(E2)
    """

    DEDUCT_ID = p + 5
    END_ID    = p + 6
    PAD_ID    = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)
    N, L = seqs.shape

    allowed_a = set(int(v) for v in a_values)
    selected_indices = []

    for i in range(N):
        if len(selected_indices) >= max_n:
            break  

        row = seqs[i]


        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

 
        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

    
        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

      
        a_tok = None
        for t in e1_tokens.tolist():
            if 0 <= t <= p - 1:  # digit
                a_tok = t
                break
        if a_tok is None or a_tok not in allowed_a:
            continue


        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue


        if e1_val != e2_val:
            selected_indices.append(i)


    if len(selected_indices) == 0:
        return seqs.new_empty((0, L))

    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=seqs.device)
    return seqs[selected_indices]



def select_E1_a_and_E1_not_eq_E2(
    seqs: torch.Tensor,
    a_values: Sequence[int],
    p: int = 11,
    max_n: int = 1000,  
) -> torch.Tensor:
    """
    Return up to max_n sequences where:
      - The *third* digit in E1 ∈ a_values
      - eval(E1) ≠ eval(E2)
    """

    DEDUCT_ID = p + 5
    END_ID    = p + 6
    PAD_ID    = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)
    N, L = seqs.shape

    allowed_a = set(int(v) for v in a_values)
    selected_indices = []

    for i in range(N):
        if len(selected_indices) >= max_n:
            break  

        row = seqs[i]

        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

    
        digits_in_e1 = [t for t in e1_tokens.tolist() if 0 <= t <= p - 1]
        if len(digits_in_e1) < 3:
            continue
        a_tok = digits_in_e1[2] 

        if a_tok not in allowed_a:
            continue

        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        if e1_val != e2_val:
            selected_indices.append(i)

    if len(selected_indices) == 0:
        return seqs.new_empty((0, L))

    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=seqs.device)
    return seqs[selected_indices]






def select_E1_not_eq_E2(
    seqs: torch.Tensor,
    p: int = 11,
    max_n: int = 1000,
) -> torch.Tensor:
    """
    Return up to max_n sequences where eval(E1) != eval(E2).

    Assumes each sequence has the structure:
        E1 DEDUCT E2 DEDUCT ... END/PAD
    and uses eval_expr_modp_1 to compute the value of E1 and E2 mod p.
    """

    DEDUCT_ID = p + 5
    END_ID    = p + 6
    PAD_ID    = p + 7

    assert seqs.ndim == 2 and seqs.dtype in (torch.long, torch.int64)
    N, L = seqs.shape

    selected_indices = []

    for i in range(N):
        if len(selected_indices) >= max_n:
            break

        row = seqs[i]

        # find first non-PAD position as start of E1
        nonpad = (row != PAD_ID).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            continue
        start = int(nonpad[0].item())

        # find positions of DEDUCT tokens
        deds = (row == DEDUCT_ID).nonzero(as_tuple=False).squeeze(-1)
        if deds.numel() < 2:
            continue

        # first DEDUCT after start → end of E1
        first_after_start = deds[deds >= start]
        if first_after_start.numel() == 0:
            continue
        first_ded = int(first_after_start[0].item())

        # second DEDUCT → end of E2
        after_first = deds[deds > first_ded]
        if after_first.numel() == 0:
            continue
        second_ded = int(after_first[0].item())

        # slice E1, E2 token spans
        if first_ded <= start:
            continue
        e1_tokens = row[start:first_ded]

        if second_ded - first_ded <= 1:
            continue
        e2_tokens = row[first_ded + 1: second_ded]

        # evaluate E1, E2
        e1_val = eval_expr_modp_1(e1_tokens, p=p)
        e2_val = eval_expr_modp_1(e2_tokens, p=p)
        if e1_val is None or e2_val is None:
            continue

        # keep only those with different values
        if e1_val != e2_val:
            selected_indices.append(i)

    if len(selected_indices) == 0:
        return seqs.new_empty((0, L))

    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=seqs.device)
    return seqs[selected_indices]





def compute_structure_accuracy_from_first_deduct_seq(
    pred_seqs: torch.Tensor,
    target_seqs: torch.Tensor,
    p: int,
) -> Dict[str, float]:

    assert pred_seqs.shape == target_seqs.shape, "pred and target must have same shape"
    device = target_seqs.device
    pred_seqs   = pred_seqs.to(device)
    target_seqs = target_seqs.to(device)

    B, T = target_seqs.shape

    PLUS_ID   = p
    MINUS_ID  = p + 1
    DEDUCT_ID = p + 5
    
    pred    =  pred_seqs
    target  =  target_seqs

    mask_from = torch.zeros_like(target, dtype=torch.bool)  # [B, T]

    for i in range(B):
        row = target[i]  # [T]
        ded_pos = (row == DEDUCT_ID).nonzero(as_tuple=False)
        if ded_pos.numel() == 0:
            continue
        first_ded = int(ded_pos[0].item())
        mask_from[i, first_ded:] = True


    is_deduct = (target == DEDUCT_ID)
    is_pm     = (target == PLUS_ID) | (target == MINUS_ID)

    struct_mask = mask_from & (is_deduct | is_pm)
    deduct_mask = mask_from & is_deduct
    pm_mask     = mask_from & is_pm

    def _safe_acc(mask: torch.Tensor) -> float:
        count = mask.sum().item()
        if count == 0:
            return float("nan")
        correct = ((pred == target) & mask).sum().item()
        return float(correct) / float(count)

    overall_acc    = _safe_acc(struct_mask)
    deduct_acc     = _safe_acc(deduct_mask)
    plusminus_acc  = _safe_acc(pm_mask)

    return overall_acc







from typing import Tuple
import torch

def proportion_E2_E3_joint_cases(
    seqs_true: torch.Tensor,
    seqs_pred: torch.Tensor,
    p: int = 11,
    e2_start_idx: int = 8,
    e2_end_idx: int = 11,
    r_idx: int = 12,
) -> Tuple[float, float, float, float]:


    assert seqs_true.ndim == 2 and seqs_true.dtype in (torch.long, torch.int64)
    assert seqs_pred.shape == seqs_true.shape, "seqs_true and seqs_pred must have the same shape"

    N, L = seqs_true.shape

    PAD_ID = p + 7

    count_den = 0

    count_E2_eq_r_eq   = 0
    count_E2_eq_r_neq  = 0
    count_E2_neq_r_eq  = 0
    count_E2_neq_r_neq = 0

   
    if not (0 <= e2_start_idx < e2_end_idx <= L and 0 <= r_idx < L):
        raise ValueError(f"Indices out of range: L={L}, "
                         f"e2_start_idx={e2_start_idx}, e2_end_idx={e2_end_idx}, r_idx={r_idx}")

    for i in range(N):
        row_true = seqs_true[i]
        row_pred = seqs_pred[i]

       
        if (row_true != PAD_ID).sum().item() == 0:
            continue

        e2_true = row_true[e2_start_idx:e2_end_idx]
        e2_pred = row_pred[e2_start_idx:e2_end_idx]

        r_true = row_true[r_idx]
        r_pred = row_pred[r_idx]

    
        if r_true == PAD_ID or r_pred == PAD_ID:
            continue

        count_den += 1

        E2_eq = torch.equal(e2_true, e2_pred)
        r_eq  = (int(r_true.item()) == int(r_pred.item()))

        if E2_eq and r_eq:
            count_E2_eq_r_eq += 1
        elif E2_eq and (not r_eq):
            count_E2_eq_r_neq += 1
        elif (not E2_eq) and r_eq:
            count_E2_neq_r_eq += 1
        else:  # not E2_eq and not r_eq
            count_E2_neq_r_neq += 1

    if count_den > 0:
        ratio_E2_eq_r_eq   = count_E2_eq_r_eq  / count_den
        ratio_E2_eq_r_neq  = count_E2_eq_r_neq / count_den
        ratio_E2_neq_r_eq  = count_E2_neq_r_eq / count_den
        ratio_E2_neq_r_neq = count_E2_neq_r_neq / count_den
    else:
        ratio_E2_eq_r_eq   = 0.0
        ratio_E2_eq_r_neq  = 0.0
        ratio_E2_neq_r_eq  = 0.0
        ratio_E2_neq_r_neq = 0.0

    return (
        ratio_E2_eq_r_eq,
        ratio_E2_eq_r_neq,
        ratio_E2_neq_r_eq,
        ratio_E2_neq_r_neq,
    )






####################extendded###################


def proportion_E2_E3_joint_cases_extended(
    seqs_true: torch.Tensor,
    seqs_pred: torch.Tensor,
    p: int = 11,
    e2_start_idx: int = 10,   
    e2_end_idx: int = 13,     
    r_idx: int = 14,     
) -> Tuple[float, float, float, float]:
    assert seqs_true.ndim == 2 and seqs_true.dtype in (torch.long, torch.int64)
    assert seqs_pred.shape == seqs_true.shape, "seqs_true and seqs_pred must have the same shape"

    N, L = seqs_true.shape
    PAD_ID = p + 7

    count_den = 0
    count_E2_eq_r_eq   = 0
    count_E2_eq_r_neq  = 0
    count_E2_neq_r_eq  = 0
    count_E2_neq_r_neq = 0

    if not (0 <= e2_start_idx < e2_end_idx <= L and 0 <= r_idx < L):
        raise ValueError(
            f"Indices out of range: L={L}, "
            f"e2_start_idx={e2_start_idx}, e2_end_idx={e2_end_idx}, r_idx={r_idx}"
        )

    for i in range(N):
        row_true = seqs_true[i]
        row_pred = seqs_pred[i]

        # 跳过全 PAD 的行
        if (row_true != PAD_ID).sum().item() == 0:
            continue

        # 取 E2 片段
        e2_true = row_true[e2_start_idx:e2_end_idx]
        e2_pred = row_pred[e2_start_idx:e2_end_idx]

        # 取 r
        r_true = row_true[r_idx]
        r_pred = row_pred[r_idx]

        # r 如果是 PAD 就跳过
        if r_true == PAD_ID or r_pred == PAD_ID:
            continue

        count_den += 1

        E2_eq = torch.equal(e2_true, e2_pred)
        r_eq  = (int(r_true.item()) == int(r_pred.item()))

        if E2_eq and r_eq:
            count_E2_eq_r_eq += 1
        elif E2_eq and (not r_eq):
            count_E2_eq_r_neq += 1
        elif (not E2_eq) and r_eq:
            count_E2_neq_r_eq += 1
        else:
            count_E2_neq_r_neq += 1

    if count_den > 0:
        ratio_E2_eq_r_eq   = count_E2_eq_r_eq  / count_den
        ratio_E2_eq_r_neq  = count_E2_eq_r_neq / count_den
        ratio_E2_neq_r_eq  = count_E2_neq_r_eq / count_den
        ratio_E2_neq_r_neq = count_E2_neq_r_neq / count_den
    else:
        ratio_E2_eq_r_eq   = 0.0
        ratio_E2_eq_r_neq  = 0.0
        ratio_E2_neq_r_eq  = 0.0
        ratio_E2_neq_r_neq = 0.0

    return (
        ratio_E2_eq_r_eq,
        ratio_E2_eq_r_neq,
        ratio_E2_neq_r_eq,
        ratio_E2_neq_r_neq,
    )







@torch.no_grad()
def measure_e2_all_digits_intervention_ce_modp_extended(
    model,
    data: torch.Tensor,                       
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of random replacements per sample
    digits_for_replace: Optional[Sequence[int]] = None,  # allowed digits for product replacement
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    seed: Optional[int] = None, 
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Evaluate robustness for SHORT layout by intervening ONLY the E2 product token.

    Differences from the "all-digits-in-E2" version:
      - We locate E2 as the span between the first and second DED markers.
      - Inside E2, we identify the *product* as the FIRST digit token in that span,
        and we only replace that single token (one-point intervention).
      - E3 (final result) for short layout is assumed at index -2 (just before END).
    """

    # Basic checks
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Special token IDs for mod p
    PLUS_ID      = p
    MINUS_ID     = p + 1
    LPAREN_ID    = p + 2
    RPAREN_ID    = p + 3
    MUL_ID       = p + 4
    DED_ID       = p + 5
    STOP_ID      = p + 6
    PAD_ID       = p + 7

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # --- Short layout: final result (E3) is at -2 (just before END) ---
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # Restrict to digit classes [0..p-1]
        return model_outputs_logits[:, -3, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # 1) Forward originals to get baseline distribution P over digits at E3
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)                     # [B, L, V]
            e3_logits = _e3_logits_from(logits)          # [B, p]
            P[s:e] = F.softmax(e3_logits, dim=-1)        # [B, p]

    # 2) Build intervention variants by replacing ONLY the E2 product token
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    # Allowed digits for replacement
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]  # [L], CPU tensor

        # Find first non-PAD index
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # Locate the two DED markers delimiting E2 in short layout
        idx_ded_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if idx_ded_all.numel() < 2:
            continue

        first_ded_candidates = idx_ded_all[idx_ded_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        second_ded_candidates = idx_ded_all[idx_ded_all > first_ded]
        if second_ded_candidates.numel() == 0:
            continue
        second_ded = int(second_ded_candidates[0].item())

        # E2 must contain at least one token
        if second_ded <= first_ded + 1:
            continue

        e2_slice = seq[first_ded + 1 : second_ded]

        # Identify the product as the SECOND digit token in E2
        rel_digit_positions = (e2_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digit_positions.numel() == 0:
            # No digit in E2; skip
            continue
        first_rel_digit = int(rel_digit_positions[0].item())
        product_abs_pos = first_ded + 1 + first_rel_digit

        # Create num_variants interventions per sample (replace ONLY the product token)
        orig_product = int(seq[product_abs_pos].item())
        for _ in range(num_variants):
            new_seq = seq.clone()

            # Build candidate set (exclude original when possible)
            if orig_product in allowed_digits and len(allowed_digits) > 1:
                candidates = [d for d in allowed_digits if d != orig_product]
            else:
                candidates = allowed_digits

            if not candidates:
                continue

            new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
            new_seq[product_abs_pos] = new_digit

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # If no interventions were possible, return zeros
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zerosN = torch.zeros(N)
        return 0.0, zerosN, 0.0, zerosN

    # Stack all modified sequences and map to their source indices
    modified_tensor = torch.stack(modified_seqs, dim=0)                                   # [M, L] CPU
    src_idx_tensor = torch.tensor(modified_src_idx, device=device, dtype=torch.long)      # [M] on device
    M = modified_tensor.shape[0]

    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_sum_inv = torch.zeros(N, device=device, dtype=torch.float32)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    # 3) Forward modified sequences to get Q and compute CE(P || Q) and invariance
    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)  # [B, L]
            logits_mod, _ = model(batch)                                 # [B, L, V]
            e3_logits_mod = _e3_logits_from(logits_mod)                  # [B, p]
            Q = F.softmax(e3_logits_mod, dim=-1)                         # [B, p]
            logQ = torch.log(Q + 1e-12)                                  # [B, p]

            ori_idx = src_idx_tensor[s:e]                                # [B]
            P_sel = P[ori_idx]                                           # [B, p]

            # Cross-entropy CE(P, Q) = - sum_k P_k log Q_k
            ce_vec = -(P_sel * logQ).sum(dim=-1)                         # [B]

            # Invariance: argmax preserved?
            inv_vec = (P_sel.argmax(dim=-1) == Q.argmax(dim=-1)).float() # [B]

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(0, ori_idx, torch.ones_like(ori_idx, dtype=torch.long))

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    # Return per-sample tensors on CPU
    return float(avg_ce_all), per_sample_avg_ce.detach().cpu(), float(inv_mean), per_sample_inv_avg.detach().cpu()






import torch
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, List

@torch.no_grad()
def measure_e1_abc_digits_intervention_ce_modp_extended(
    model,
    data: torch.Tensor,                       # [N, L] int64 tokens
    p: int,                                   # modulus p (digits are 0..p-1)
    num_variants: int = 50,                   # number of variants per sample
    digits_for_replace: Optional[Sequence[int]] = None,
    eval_batch_size: int = 512,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    E1 只干预 c 的版本（适配新结构）:

      新数据布局: (a ± b) * c ± d -> s ± d -> r
      tokens:
        0:  LP
        1:  a
        2:  op1
        3:  b
        4:  RP
        5:  MUL
        6:  c
        7:  op2
        8:  d
        9:  DED
        10: s
        11: op2
        12: d
        13: DED
        14: r   (E3)
        15: END

    干预策略：
      - 在 E1 段里找到前三个 digit，把它们视为 a、b、c，
        但只随机替换第三个 digit（c）。
      - 观察最终输出分布 (r 位置) 在干预前后的变化：
          * 平均 cross-entropy CE(P, Q)
          * argmax 是否保持不变的比例 (invariance)
    """

    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), \
        "data must be [N, L] int64"
    N, L = data.shape
    data = data.clone()

    # Token IDs
    DED_ID = p + 5
    PAD_ID = p + 7

    # Device resolution
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    if seed is not None:
        torch.manual_seed(seed)

    # ---- E3 logits: 按新结构固定在 index 14（r 的位置） ----
    def _e3_logits_from(model_outputs_logits: torch.Tensor) -> torch.Tensor:
        # 假定所有样本长度都是 16，且 r 在 index 14
        return model_outputs_logits[:, 14, :][:, :p]  # [B, p]

    was_training = model.training
    model.eval()

    # === 1) baseline P ===
    with torch.no_grad():
        P = torch.empty((N, p), device=device, dtype=torch.float32)
        for s in range(0, N, eval_batch_size):
            e = min(s + eval_batch_size, N)
            batch = data[s:e].to(device, non_blocking=True)
            logits, _ = model(batch)
            e3_logits = _e3_logits_from(logits)
            P[s:e] = F.softmax(e3_logits, dim=-1)

    # === 2) 构造干预样本（只干预 E1 中的 c） ===
    modified_seqs: List[torch.Tensor] = []
    modified_src_idx: List[int] = []

    # 允许替换的 digit
    if digits_for_replace is None:
        if p > 5:
            allowed_digits = [d for d in range(p) if d not in (0, 5)]
        else:
            allowed_digits = [d for d in range(p) if d != 0]
    else:
        allowed_digits = list(digits_for_replace)

    for i in range(N):
        seq = data[i]

        # 找第一个非 PAD 位置
        nonpad_idx = (seq != PAD_ID).nonzero(as_tuple=False)
        if nonpad_idx.numel() == 0:
            continue
        start_idx = int(nonpad_idx[0].item())

        # 找第一个 DED（E1 / E2 的分隔）
        ded_idx_all = (seq == DED_ID).nonzero(as_tuple=False).squeeze(-1)
        if ded_idx_all.numel() == 0:
            continue
        first_ded_candidates = ded_idx_all[ded_idx_all >= start_idx]
        if first_ded_candidates.numel() == 0:
            continue
        first_ded = int(first_ded_candidates[0].item())

        # E1 = [start_idx, first_ded)
        e1_slice = seq[start_idx:first_ded]

        # 找 E1 中所有 digit 位置
        rel_digits = (e1_slice < p).nonzero(as_tuple=False).squeeze(-1)
        if rel_digits.numel() < 3:
            # 少于三个 digit，无法定位 a,b,c
            continue

        # a,b,c = E1 中前三个 digit；这里只用第三个，即 c
        rel_c = rel_digits[2]
        c_abs_pos = int(rel_c.item()) + start_idx

        # 对每个样本构造 num_variants 个干预版本（只改 c）
        orig_c = int(seq[c_abs_pos].item())
        for _ in range(num_variants):
            new_seq = seq.clone()

            if len(allowed_digits) > 1:
                candidates = [d for d in allowed_digits if d != orig_c]
            else:
                candidates = allowed_digits

            if not candidates:
                continue

            new_digit = candidates[torch.randint(0, len(candidates), (1,)).item()]
            new_seq[c_abs_pos] = new_digit

            modified_seqs.append(new_seq)
            modified_src_idx.append(i)

    # === 3) 计算 CE & invariance ===
    if len(modified_seqs) == 0:
        if was_training:
            model.train()
        zeros = torch.zeros(N)
        return 0.0, zeros, 0.0, zeros

    modified_tensor = torch.stack(modified_seqs, dim=0)          # [M, L]
    src_idx_tensor = torch.tensor(modified_src_idx,
                                  device=device, dtype=torch.long)
    M = modified_tensor.shape[0]

    total_ce = 0.0
    total_inv = 0.0
    count = 0

    per_sample_sum_ce  = torch.zeros(N, device=device)
    per_sample_sum_inv = torch.zeros(N, device=device)
    per_sample_cnt     = torch.zeros(N, device=device, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, M, eval_batch_size):
            e = min(s + eval_batch_size, M)
            batch = modified_tensor[s:e].to(device, non_blocking=True)
            logits_mod, _ = model(batch)
            e3_logits_mod = _e3_logits_from(logits_mod)      # [B, p]

            logQ = F.log_softmax(e3_logits_mod, dim=-1)
            q_argmax = e3_logits_mod.argmax(dim=-1)

            ori_idx = src_idx_tensor[s:e]
            P_sel = P[ori_idx]
            p_argmax = P_sel.argmax(dim=-1)

            ce_vec = -(P_sel * logQ).sum(dim=-1)
            inv_vec = (p_argmax == q_argmax).float()

            total_ce  += ce_vec.sum().item()
            total_inv += inv_vec.sum().item()
            count     += ce_vec.numel()

            per_sample_sum_ce.index_add_(0, ori_idx, ce_vec)
            per_sample_sum_inv.index_add_(0, ori_idx, inv_vec)
            per_sample_cnt.index_add_(
                0, ori_idx,
                torch.ones_like(ori_idx, dtype=torch.long)
            )

    avg_ce_all = total_ce / max(count, 1)
    inv_mean   = total_inv / max(count, 1)

    per_sample_avg_ce  = per_sample_sum_ce  / per_sample_cnt.clamp_min(1)
    per_sample_inv_avg = per_sample_sum_inv / per_sample_cnt.clamp_min(1)

    if was_training:
        model.train()

    return (
        float(avg_ce_all),
        per_sample_avg_ce.detach().cpu(),
        float(inv_mean),
        per_sample_inv_avg.detach().cpu(),
    )




@torch.no_grad()
def compute_e3_entropy_sharpness_modp_extended(
    model,
    data: torch.Tensor,               
    p: int,                           
    eval_batch_size: int = 512, 
    device: Optional[torch.device] = None, 
    e3_logits_pos: int = -3,         
) -> Tuple[torch.Tensor, float]:
    """
    Entropy-based sharpness at E3 (short layout), ignoring ground truth:
      sharpness = exp(H(P_digits)), where P_digits is the softmax over digit classes [0..p-1]
      taken from the E3 logits position (default -3).

    Returns:
      per_sample_sharp: [N] tensor, exp(H) per sample (CPU)
      sharp_mean:       float, mean exp(H) across samples
    """
    assert data.dim() == 2 and data.dtype in (torch.long, torch.int64), "data must be [N, L] int64"
    N, L = data.shape

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = data.device

    was_training = model.training
    model.eval()

    per_sample_sharp = torch.empty(N, dtype=torch.float32)

    eps = 1e-12
    for s in range(0, N, eval_batch_size):
        e = min(s + eval_batch_size, N)
        batch = data[s:e].to(device, non_blocking=True)      # [B, L]

        # Forward pass
        out = model(batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out   # [B, L, V]

        # Take E3 logits at -3 and restrict to digit classes [0..p-1]
        e3_logits = logits[:, e3_logits_pos, :][:, :p]       # [B, p]

        # P_digits: distribution over digits only
        P = F.softmax(e3_logits, dim=-1)                     # [B, p]

        # Shannon entropy H(P) over digits and entropy-based sharpness exp(H)
        H = -(P * (P.clamp_min(eps).log())).sum(dim=-1)      # [B]
        sharp = torch.exp(H).float().cpu()                   # [B]

        per_sample_sharp[s:e] = sharp

    sharp_mean = float(per_sample_sharp.mean().item())

    if was_training:
        model.train()

    return per_sample_sharp, sharp_mean