# importing required libraries
import os
import warnings
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Sequence
from arithmetic_experiments.extended_task_experiment.utils import matching_scores, TrainingInfo, autoregressive_extend, induction_score, simplification_head_score, eval_opposite_structure
from tasks.extended_task.modular_data_generation_extended import generate_noisy_twosteps_dataset_modp, generate_noisy_twosteps_dataset_modp_1, ModularVocabulary,compute_e1_e2_distributions, compute_short_E3_distributions, generate_noisy_twosteps_varied_dataset_modp, generate_noisy_twosteps_dataset_modp_extended
from pathlib import Path
from arithmetic_experiments.extended_task_experiment.unfaithfulness_metrics_extended import measure_e2_all_digits_intervention_ce_modp, measure_e2_all_digits_intervention_ce_modp_short, measure_e1_ab_digits_intervention_ce_modp_short, compute_e3_entropy_sharpness_modp_short,measure_e3_agreement_with_e1_e2_modp, autoreg_generate_from_firstded_to_stop_1, autoreg_generate_from_secondded_to_stop_1, proportion_E1_eq_r_but_E2_neq_r_1,third_layer_attn_ffn_no_residual, shuffle_e2_first_digit_modp_short, get_last_layer_QK, select_E1_a_in_firstN, proportion_E2_eq_r_1, proportion_E1_eq_r_1, select_E1_a_and_E1_not_eq_E2, select_E1_not_eq_E2, measure_e2_addup_intervention_ce_modp_short, proportion_E1_eq_r_but_E2_neq_r_1, proportion_E1_neq_r_but_E2_eq_r_1, proportion_E1_eq_r_and_E2_eq_r_1, proportion_E1_neq_r_and_E2_neq_r_1, compute_structure_accuracy_from_first_deduct_seq, proportion_E2_E3_joint_cases, proportion_E2_E3_joint_cases_extended, measure_e2_all_digits_intervention_ce_modp_extended, measure_e1_abc_digits_intervention_ce_modp_extended, compute_e3_entropy_sharpness_modp_extended



def build_token_display_map(modulus: int, vocab: ModularVocabulary | dict | None = None) -> dict:  ## CHANGED
    """Create a decoder map for token ids supporting general modulus."""  ## CHANGED

    if isinstance(vocab, ModularVocabulary):  ## CHANGED
        vocab_tokens = vocab.tokens  ## CHANGED
    elif isinstance(vocab, dict):  ## CHANGED
        vocab_tokens = vocab  ## CHANGED
    else:  ## CHANGED
        vocab_tokens = None  ## CHANGED
    if vocab_tokens is not None:  ## CHANGED
        numeric_tokens = [int(k) for k in vocab_tokens.keys() if k.isdigit()]  ## CHANGED
        if numeric_tokens:  ## CHANGED
            modulus = max(numeric_tokens) + 1  ## CHANGED

    special_order = {  ## CHANGED
        "PLUS": 0,
        "MINUS": 1,
        "LP": 2,
        "RP": 3,
        "MUL": 4,
        "DEDUCT": 5,
        "END": 6,
        "PAD": 7,
    }  ## CHANGED

    render_map = {  ## CHANGED
        "PLUS": "+",
        "MINUS": "-",
        "LP": "(",
        "RP": ")",
        "MUL": "*",
        "DEDUCT": "-->",
        "END": "#",
        "PAD": " ",
    }  ## CHANGED

    token_map = {}  ## CHANGED
    for name, offset in special_order.items():  ## CHANGED
        if vocab_tokens is not None and name in vocab_tokens:  ## CHANGED
            token_map[vocab_tokens[name]] = render_map[name]  ## CHANGED
        else:  ## CHANGED
            token_map[modulus + offset] = render_map[name]  ## CHANGED

    return token_map  ## CHANGED


def decode_expressions(tensor_batch, modulus: int = 11, token_map: dict | None = None):  ## CHANGED
    if token_map is None:  ## CHANGED
        token_map = build_token_display_map(modulus)  ## CHANGED

    decoded = []  ## CHANGED
    for row in tensor_batch:  ## CHANGED
        expr = []  ## CHANGED
        for token in row:  ## CHANGED
            if isinstance(token, torch.Tensor):  ## CHANGED
                token = token.item()  ## CHANGED
            if 0 <= token < modulus:  ## CHANGED
                expr.append(str(token))  ## CHANGED
            elif token in token_map:  ## CHANGED
                expr.append(token_map[token])  ## CHANGED
            else:  ## CHANGED
                expr.append('?')  ## CHANGED
        decoded.append(''.join(expr))  ## CHANGED
    return decoded  ## CHANGED



def make_scheduler(optimizer, config):
    if config.schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif config.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epoch, config.lr_min
        )
    return scheduler


def get_loss(model, criterion, src: torch.Tensor, mask=None):
    """
    calculates the loss for a train batch to prepare for backpropagation
    """
    output,_ = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )

    if mask is not None: # requires criterion (e.g., nn.CrossEntropyLoss) to be unreduced

        loss = torch.sum(loss * mask[:,:-1].contiguous().view(-1)) / torch.sum(mask[:, :-1])

    return loss




def get_weighted_loss(
    model,
    src: torch.Tensor,                 # [B, T]
    sample_weights: torch.Tensor,      
    total_N: int=1383680,                      
    mask: torch.Tensor = None,         # [B, T]，1/0
    label_smoothing: float = 0.0,
    sampling_mode: str = "uniform",    # "uniform" or "weighted_by_w"
):
    logits, _ = model(src)                   # [B, T, V]
    B, T, V = logits.shape

    # next-token shift
    logits = logits[:, :-1, :].contiguous().view(-1, V)  # [(B*(T-1)), V]
    targets = src[:, 1:].contiguous().view(-1)           # [(B*(T-1))]

    per_tok_loss = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )  # [(B*(T-1))]

    # position mask
    if mask is not None:
        pos_mask = mask[:, :-1].to(per_tok_loss.device).float()  # [B, T-1]
    else:
        pos_mask = torch.ones((B, T-1), device=per_tok_loss.device, dtype=torch.float)
    sw = sample_weights.to(per_tok_loss.device).float().unsqueeze(1).expand(B, T-1)  # [B, T-1]

    if sampling_mode == "uniform":
        eff_w = (sw * pos_mask).reshape(-1)                # [(B*(T-1))]
        loss_sum = torch.sum(per_tok_loss * eff_w)        
        loss = (total_N / B) * loss_sum

    elif sampling_mode == "weighted_by_w":
        tok_loss = per_tok_loss.view(B, T-1)
        denom = pos_mask.sum(dim=1).clamp_min(1.0)       
        per_sample_ce = (tok_loss * pos_mask).sum(dim=1) / denom  # [B]
        loss = per_sample_ce.mean()
    else:
        raise ValueError("sampling_mode must be 'uniform' or 'weighted_by_w'")

    return loss








def get_mask(start_ind_list: List, max_len: int) -> torch.Tensor:
    """
    get_mask returns a mask for a train/test batch
        to mask out a number of initial tokens in each sequence
    Args:7777786677771
        start_ind_list: a list that contains the index of each sequence from which prediction accurary counts
        max_len: the maximum sequence length
    """
    n = len(start_ind_list)

    mask = torch.ones((n, max_len), dtype=torch.long)  # Initialize with ones
    for i, l in enumerate(start_ind_list):
        mask[i, :(l-1)] = 0  # Set the first L[i] elements to zero in each row
    return mask



@torch.no_grad()
def loss_err(model, criterion, src, mask=None):
    """
    Calculates the loss, sentence-level error, element-wise error, and predictions
    for evaluation purposes.
    """
    model.eval()
    output, _ = model(src)
    vocab_size = output.size(-1)

    # 1) Loss
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )
    if mask is not None:
        flat_mask = mask[:, :-1].contiguous().view(-1)
        loss = (loss * flat_mask).sum() / flat_mask.sum()

    # 2) Predictions (shifted)
    pred = output.argmax(dim=2)[:, :-1]  # (B, T-1)
    target = src[:, 1:]                 # (B, T-1)

    # 3) Sentence-level error
    if mask is not None:
        valid = mask[:, :-1].bool()
        correct_sent = (pred == target) | (~valid)
        sent_err = 1 - correct_sent.all(dim=1).float().mean()
    else:
        sent_err = 1 - pred.eq(target).all(dim=1).float().mean()

    # 4) Element-wise error
    if mask is not None:
        correct_tok = (pred == target).float() * mask[:, :-1].float()
        elem_err = 1 - correct_tok.sum() / mask[:, :-1].float().sum()
    else:
        elem_err = 1 - pred.eq(target).float().mean()

    
    if pred.size(1) < 2:
        last2_acc = torch.tensor(0.0, device=src.device)
    else:
        last2_acc = (pred[:, -2] == target[:, -2]).float().mean()


    if pred.size(1) >= 5:
        last4_acc = (pred[:, -5:-1] == target[:, -5:-1]).float().mean()
    else:
        last4_acc = torch.tensor(0.0, device=pred.device)


    return loss, sent_err, elem_err, pred.cpu(), last2_acc, last4_acc




def compute_sharpness(model, criterion, src, rho=1e-2, num_samples=5, mask=None):

    device = next(model.parameters()).device
    model.eval()

    # Compute original loss
    with torch.no_grad():
        base_loss, _, _, _, _, _ = loss_err(model, criterion, src.to(device), mask)
        base_loss = base_loss.item()

    max_loss = -float('inf')

    for _ in range(num_samples):
        # Copy model and apply perturbation to parameters
        perturbed_model = copy.deepcopy(model)
        for param in perturbed_model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * rho
                param.data.add_(noise)

        # Compute loss with perturbed weights
        with torch.no_grad():
            loss, _, _, _, _, _  = loss_err(perturbed_model, criterion, src.to(device), mask)
            loss = loss.item()
            max_loss = max(max_loss, loss)

    # Sharpness formula
    sharpness = (max_loss - base_loss) / (1.0 + base_loss)
    return sharpness
















@torch.no_grad()
def loss_err_weighted(
    model,
    src: torch.Tensor,                 # [B, T]
    sample_weights: torch.Tensor=None, 
    total_N: int = 1_383_680,         
    mask: torch.Tensor = None,         # [B, T] 1/0
    label_smoothing: float = 0.0,
    sampling_mode: str = "uniform",    # "uniform" or "weighted_by_w"
):
    model.eval()
    logits, _ = model(src)         # [B, T, V]
    B, T, V = logits.shape

    # Shift for next-token prediction
    logits_shift = logits[:, :-1, :].contiguous().view(-1, V)   # [(B*(T-1)), V]
    targets = src[:, 1:].contiguous().view(-1)                  # [(B*(T-1))]

    # Per-token CE (no reduction)
    per_tok_loss = F.cross_entropy(
        logits_shift, targets, reduction="none", label_smoothing=label_smoothing
    )  # [(B*(T-1))]

    # Position mask
    if mask is not None:
        pos_mask = mask[:, :-1].to(per_tok_loss.device).float()  # [B, T-1]
    else:
        pos_mask = torch.ones((B, T-1), device=per_tok_loss.device, dtype=torch.float)

    if sampling_mode == "uniform":
        if sample_weights is None:
            raise ValueError("sampling_mode='uniform' need sample_weights=[B].")
        sw = sample_weights.to(per_tok_loss.device).float().unsqueeze(1).expand(B, T-1)  # [B, T-1]
        eff_w = (sw * pos_mask).reshape(-1)                    # [(B*(T-1))]
        loss_sum = torch.sum(per_tok_loss * eff_w)
        loss = (total_N / B) * loss_sum

    elif sampling_mode == "weighted_by_w":

        tok_loss = per_tok_loss.view(B, T-1)
        denom = pos_mask.sum(dim=1).clamp_min(1.0)           
        per_sample_ce = (tok_loss * pos_mask).sum(dim=1) / denom  # [B]
        loss = per_sample_ce.mean()

    else:
        raise ValueError("sampling_mode must be 'uniform' or 'weighted_by_w'")

    # --------- Predictions for error metrics ---------
    pred = logits.argmax(dim=2)[:, :-1]   # (B, T-1)
    target = src[:, 1:]                   # (B, T-1)

    # Sentence-level error
    if mask is not None:
        valid = mask[:, :-1].bool()
        correct_sent = (pred == target) | (~valid)
        sent_err = 1.0 - correct_sent.all(dim=1).float().mean()
    else:
        sent_err = 1.0 - pred.eq(target).all(dim=1).float().mean()

    # Element-wise error
    if mask is not None:
        correct_tok = (pred == target).float() * mask[:, :-1].float()
        elem_err = 1.0 - correct_tok.sum() / mask[:, :-1].float().sum()
    else:
        elem_err = 1.0 - pred.eq(target).float().mean()

    # Last-2 accuracy
    if pred.size(1) < 2:
        last2_acc = torch.tensor(0.0, device=src.device)
    else:
        last2_acc = (pred[:, -2] == target[:, -2]).float().mean()

    # Last-4 (actually last 4 tokens excluding EOS shift) accuracy over window length 4
    if pred.size(1) >= 5:
        last4_acc = (pred[:, -5:-1] == target[:, -5:-1]).float().mean()
    else:
        last4_acc = torch.tensor(0.0, device=pred.device)

    return loss, sent_err, elem_err, pred.cpu(), last2_acc, last4_acc






def save_checkpoints(
    config,
    epoch,
    src_test,
    model,
    optimizer,
    scheduler,
    record_loss,
    record_test_error,
    record_train_error,
    save_lr = False
):

    with torch.no_grad():
        _, attentions = model(src_test)

    checkpoint = {'epoch': epoch,
                  'train_error': record_train_error,
                  'train_loss': record_loss[0],
                  'test_error': record_test_error[1],
                  'test_loss': record_loss[1],
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'attentions': attentions}
    #attention is a list in which there are (num_layers) tuples, each tuple stores attn and QK_vals
    if save_lr is True:
        checkpoint["learning rate"] = scheduler.get_last_lr()[0]

    ckpt_path = os.path.join(config.out_dir, "MA_checkpoints")
    out_path = os.path.join(ckpt_path,
                            f"ckpt_{epoch + 1}_{(1 + epoch) // (config.num_epoch // config.n_save)}.pt")

    torch.save(checkpoint, out_path)
    print(f"Checkpoint {(1 + epoch) // (config.num_epoch // config.n_save)}/{config.n_save} saved")


    
    
    
def estimate_E_vHv_gaussian(
    model,
    criterion,
    src: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_samples: int = 20,
    sigma: float = 1.0,              # std of Gaussian: v ~ N(0, sigma^2 I)
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = src.device

    model.eval()
    params: Iterable[torch.nn.Parameter] = [
        p for p in model.parameters() if p.requires_grad
    ]
    params = list(params)

    # Move inputs to device
    src = src.to(device, non_blocking=True)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    vHv_list = []

    for _ in range(num_samples):
        # 1) Compute loss and gradient g = dL/dθ (create_graph=True for Hessian)
        loss = get_loss(model, criterion, src, mask=mask)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,   
            retain_graph=True    
        )

        # 2) Sample Gaussian vector v for each parameter tensor
        vs = [sigma * torch.randn_like(p_i, device=device) for p_i in params]

        # 3) Compute g·v = sum_i <g_i, v_i>
        gv = torch.zeros((), device=device)
        for g_i, v_i in zip(grads, vs):
            gv = gv + (g_i * v_i).sum()

        # 4) Hv = grad(g·v, θ)  -> Hessian-vector product
        Hv = torch.autograd.grad(
            gv,
            params,
            retain_graph=True,   
            create_graph=False
        )

        # 5) v^T H v = sum_i <v_i, Hv_i>
        vHv = torch.zeros((), device=device)
        for v_i, hv_i in zip(vs, Hv):
            vHv = vHv + (v_i * hv_i).sum()

        vHv_list.append(vHv.detach())

    vHv_values = torch.stack(vHv_list, dim=0)  # shape: (num_samples,)
    mean_vHv = vHv_values.mean()               # scalar tensor
    return mean_vHv


def compute_sam_sharpness(
    model,
    criterion,
    src: torch.Tensor,         
    mask: torch.Tensor = None, 
    rho: float = 1e-2,
    device: Optional[torch.device] = None,
) -> float:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = src.device

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    for p in params:
        if p.grad is not None:
            p.grad = None

    src = src.to(device, non_blocking=True)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    loss_orig = get_loss(model, criterion, src, mask=mask)
    grads = torch.autograd.grad(loss_orig, params, create_graph=False, retain_graph=False)

    grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
    if grad_norm == 0.0 or not math.isfinite(grad_norm):
        return 0.0

    scale = rho / (grad_norm + 1e-12)

    eps_list = []
    with torch.no_grad():
        for p, g in zip(params, grads):
            eps = g * scale
            p.add_(eps)
            eps_list.append(eps)

    loss_pert = get_loss(model, criterion, src, mask=mask)

    with torch.no_grad():
        for p, eps in zip(params, eps_list):
            p.sub_(eps)

    sharpness = (loss_pert - loss_orig).item()
    return float(max(sharpness, 0.0))  



def compute_hessian_max_eig(
    model,
    criterion,
    src: torch.Tensor,             # [B, T]
    mask: torch.Tensor = None,     # [B, T] or None
    num_iters: int = 10,
    tol: float = 1e-6,
) -> float:
    device = next(model.parameters()).device
    model_was_training = model.training
    model.train()  

    src = src.to(device, non_blocking=True)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        return float("nan")

    for p in params:
        if p.grad is not None:
            p.grad = None

    loss = get_loss(model, criterion, src, mask=mask)  
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    dim = flat_grad.numel()
    if dim == 0:
        return float("nan")

    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)

    lambda_old = 0.0

    for _ in range(num_iters):

        hvp_input = 0.0
        idx = 0
        for g in grads:
            g_flat = g.reshape(-1)
            hvp_input = hvp_input + (g_flat * v[idx: idx + g_flat.numel()]).sum()
            idx += g_flat.numel()

        # hv = ∇_θ (g·v)
        hv = torch.autograd.grad(
            hvp_input,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        flat_hv = torch.cat([h.reshape(-1) for h in hv])

        v_norm = flat_hv.norm() + 1e-12
        v = flat_hv / v_norm
        lambda_new = (v * flat_hv).sum().item()  # v^T H v

        if abs(lambda_new - lambda_old) / (abs(lambda_old) + 1e-12) < tol:
            break
        lambda_old = lambda_new

    del grads, flat_grad, hv, flat_hv, v

    if model_was_training:
        model.train()
    else:
        model.eval()

    return float(max(lambda_old, 0.0))





@torch.no_grad()
def batch_kl_mean(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape}, q {q.shape}")
    
    
    # Clamp to avoid log(0)
    p_clamped = p.clamp(min=eps)
    q_clamped = q.clamp(min=eps)
    
    # KL(p || q) for each sample
    kl_per_sample = (p_clamped * (torch.log(p_clamped) - torch.log(q_clamped))).sum(dim=-1)
    
    # Mean over batch
    return kl_per_sample.mean()


@torch.no_grad()
def e1e2_indicator_target(
    seqs: torch.Tensor,
    p: int
) -> torch.Tensor:

    if seqs.ndim != 2:
        raise ValueError("seqs must be a 2D tensor [N, L].")

    N, L = seqs.shape
    device = seqs.device
    probs = torch.full((N, p), 1.0 / p, dtype=torch.float32, device=device)

    PLUS_ID = p
    MINUS_ID = p+1

    # Short layout: (a * b) +/- c
    if L == 14:
        # E1: (a * b) +/- c
        a1 = seqs[:, 1]
        b1 = seqs[:, 3]
        op1 = seqs[:, 5]
        c1 = seqs[:, 6]

        prod1 = (a1 * b1) % p
        is_plus1 = (op1 == PLUS_ID)
        is_minus1 = (op1 == MINUS_ID)
        e1 = torch.where(
            is_plus1, (prod1 + c1) % p,
            torch.where(is_minus1, (prod1 - c1) % p, torch.full_like(prod1, -1))
        )

        # E2: product +/- c  (using positions after first DEDUCT)
        prod2 = seqs[:, 8]
        op2 = seqs[:, 9]
        c2 = seqs[:, 10]

        is_plus2 = (op2 == PLUS_ID)
        is_minus2 = (op2 == MINUS_ID)
        e2 = torch.where(
            is_plus2, (prod2 + c2) % p,
            torch.where(is_minus2, (prod2 - c2) % p, torch.full_like(prod2, -1))
        )

    # Extended layout: (a +/- b) * c +/- d
    elif L == 16:
        # E1: (a +/- b) * c +/- d
        a = seqs[:, 1]
        op1 = seqs[:, 2]
        b = seqs[:, 3]
        c = seqs[:, 6]
        op2 = seqs[:, 7]
        d1 = seqs[:, 8]

        is_plus1 = (op1 == PLUS_ID)
        is_minus1 = (op1 == MINUS_ID)
        first_partial = torch.where(
            is_plus1, (a + b) % p,
            torch.where(is_minus1, (a - b) % p, torch.full_like(a, -1))
        )

        product1 = (first_partial * c) % p

        is_plus2 = (op2 == PLUS_ID)
        is_minus2 = (op2 == MINUS_ID)
        e1 = torch.where(
            is_plus2, (product1 + d1) % p,
            torch.where(is_minus2, (product1 - d1) % p, torch.full_like(product1, -1))
        )

        # E2: product +/- d (using positions after first DEDUCT)
        prod2 = seqs[:, 10]
        op2b = seqs[:, 11]
        d2 = seqs[:, 12]

        is_plus2b = (op2b == PLUS_ID)
        is_minus2b = (op2b == MINUS_ID)
        e2 = torch.where(
            is_plus2b, (prod2 + d2) % p,
            torch.where(is_minus2b, (prod2 - d2) % p, torch.full_like(prod2, -1))
        )
    else:
        raise ValueError(
            f"Unsupported sequence length L={L}. "
            "Expected 14 (short) or 16 (extended) for this generator."
        )

    # Where E1 == E2, override with one-hot at E2
    equal_mask = (e1 == e2) & (e2 >= 0)  # safety check on invalid ops

    if equal_mask.any():
        probs[equal_mask] = 0.0
        probs[equal_mask, e2[equal_mask]] = 1.0

    return probs







def get_last_qkv_weight_param_names(model: nn.Module) -> List[str]:
    """
    Collect names of Q/K/V weight parameters ONLY in the last attention block (h_2.mha).
    Assumes parameter names contain:
        'h_2.mha.W_q.weight', 'h_2.mha.W_k.weight', 'h_2.mha.W_v.weight'.
    """
    qkv_names = []
    for name, p in model.named_parameters():
        if (
            "h_2.mha.W_q.weight" in name
            or "h_2.mha.W_k.weight" in name
            or "h_2.mha.W_v.weight" in name
        ):
            qkv_names.append(name)
    return qkv_names


def get_last_qkv_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Collect the actual Parameter tensors of Q/K/V weights
    ONLY in the last attention block (h_2.mha).
    """
    params = []
    for name, p in model.named_parameters():
        if (
            "h_2.mha.W_q.weight" in name
            or "h_2.mha.W_k.weight" in name
            or "h_2.mha.W_v.weight" in name
        ):
            if p.requires_grad:
                params.append(p)
    return params



def hessian_vector_product_subspace(
    model: nn.Module,
    criterion: nn.Module,
    src: torch.Tensor,
    mask: torch.Tensor,
    params_subspace: List[torch.nn.Parameter],
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hessian-vector product H v in the given parameter subspace.

    Args:
        model, criterion, src, mask: same as training.
        params_subspace: list of parameters where we define the Hessian.
        vector: 1D tensor (flattened) of the same total size as params_subspace.

    Returns:
        Hv_flat: 1D tensor (flattened) = H v in that subspace.
    """
    device = next(model.parameters()).device
    src = src.to(device)
    if mask is not None:
        mask = mask.to(device)

    # Make sure model is in eval mode (no dropout etc.) for sharpness measurement
    model.eval()

    # 1. Zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # 2. Compute loss and first-order grads w.r.t. params_subspace
    loss = get_loss(model, criterion, src, mask)
    grads = torch.autograd.grad(
        loss,
        params_subspace,
        create_graph=True,   # keep graph for second-order derivative
        retain_graph=True,
    )

    # 3. Flatten grads and compute inner product g^T v
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    assert flat_grads.numel() == vector.numel()
    inner = torch.dot(flat_grads, vector)

    # 4. Hessian-vector product = ∇(g^T v)
    Hv = torch.autograd.grad(
        inner,
        params_subspace,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv]).detach()
    return Hv_flat








def hessian_top_eigenvalue_last_qkv(
    model: nn.Module,
    criterion: nn.Module,
    src: torch.Tensor,
    mask: torch.Tensor = None,
    num_iters: int = 20,
) -> float:
    """
    Estimate the largest eigenvalue of the Hessian restricted to
    the last-layer Q/K/V weight subspace (h_2.mha).

    Uses power iteration on Hessian-vector products.

    Returns:
        lambda_max (float): estimated top eigenvalue.
    """
    device = next(model.parameters()).device
    src = src.to(device)
    if mask is not None:
        mask = mask.to(device)

    # 1. Collect subspace parameters (last-layer Q/K/V weights)
    params_subspace = get_last_qkv_params(model)
    if len(params_subspace) == 0:
        return 0.0

    # 2. Initialize a random vector v in that subspace
    total_dim = sum(p.numel() for p in params_subspace)
    v = torch.randn(total_dim, device=device)

    lambda_est = 0.0

    for _ in range(num_iters):
        # Normalize v
        v = v / (v.norm(p=2) + 1e-12)

        # Compute Hv
        Hv = hessian_vector_product_subspace(
            model, criterion, src, mask, params_subspace, v
        )

        # Rayleigh quotient: v^T H v
        lambda_est = torch.dot(v, Hv).item()

        # Update v for next iteration
        v = Hv.detach()

    return max(lambda_est, 0.0)  # clip small negatives due to numerical noise











def compute_sharpness_last_qkv_only(
    model: nn.Module,
    criterion: nn.Module,
    src: torch.Tensor,
    epsilon: float,
    mask: torch.Tensor = None,
) -> float:

    device = next(model.parameters()).device
    src = src.to(device)
    if mask is not None:
        mask = mask.to(device)

    model.eval()

    # 1. Save original parameters so we can restore them later
    orig_params: Dict[str, torch.Tensor] = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    # 2. Zero grads and compute base loss + gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    base_loss = get_loss(model, criterion, src, mask)
    base_loss.backward()

    # 3. Collect gradients ONLY for last-layer Q/K/V weights
    qkv_names = get_last_qkv_weight_param_names(model)

    grads_flat = []
    params_to_perturb = []

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in qkv_names and p.grad is not None:
                grads_flat.append(p.grad.view(-1))
                params_to_perturb.append((name, p))

        if len(grads_flat) == 0:
            # No last-layer QKV parameters or no gradients: no sharpness signal
            return 0.0

        flat_grad = torch.cat(grads_flat, dim=0)
        grad_norm = torch.norm(flat_grad, p=2)

        if grad_norm.item() == 0.0:
            # Zero gradient in this subspace => no sharpness
            return 0.0

        # 4. Build perturbation vector with L2 norm = epsilon (only in last-layer QKV subspace)
        flat_delta = epsilon * flat_grad / grad_norm

        # 5. Apply perturbation to each last-layer Q/K/V weight parameter
        offset = 0
        for name, p in params_to_perturb:
            numel = p.numel()
            delta_chunk = flat_delta[offset: offset + numel].view_as(p)
            p.add_(delta_chunk)  # in-place perturbation
            offset += numel

    # 6. Compute loss at perturbed parameters (no grad needed)
    with torch.no_grad():
        perturbed_loss = get_loss(model, criterion, src, mask)

    # 7. Restore original parameters
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(orig_params[name])

    sharpness = (perturbed_loss - base_loss).item()
    return sharpness









def train_with_dataloader(
    model, config, optimizer, scheduler, run,
    train_loader, N=1000, p1=0.05, p2=0.05, training_info=None, modulus: int = 11, vocab: ModularVocabulary | dict | None = None  ## CHANGED
):

    test_set, test_vocab, test_set_info = generate_noisy_twosteps_dataset_modp_extended(
        N, modulus, p1, p2, device=config.device, extended_expression = True
    )  

    
    if vocab is None:  ## CHANGED
        vocab = test_vocab  ## CHANGED
    token_map = build_token_display_map(modulus, vocab)  ## CHANGED

    test_set = test_set.long().to(config.device)  ## CHANGED
    test_seq_len = test_set.size(1)  ## CHANGED
    if hasattr(config, "max_seq_len") and test_seq_len < getattr(config, "max_seq_len"):  ## CHANGED
        warnings.warn(  ## CHANGED
            f"Test sequence length {test_seq_len} is smaller than configured max_seq_len {config.max_seq_len}."  ## CHANGED
        )  ## CHANGED
    test_mask = get_mask(test_set_info["solution_start_ind"], test_seq_len).to(config.device)  ## CHANGED

    loss_reduction = "none" if config.train_mask else "mean"
    criterion = (nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                     reduction=loss_reduction)
                 if config.label_smoothing > 0
                 else nn.CrossEntropyLoss(reduction=loss_reduction))

    if training_info is None:
        training_info = TrainingInfo()

    total_epochs = len(train_loader)
    model.train()

    record_loss = [float('nan'), float('nan')]  ## CHANGED
    record_error = [float('nan')] * 5  ## CHANGED
    test_pred = None  ## CHANGED
    result_acc = torch.tensor(float('nan'), device=config.device)  ## CHANGED
    last4_acc = torch.tensor(float('nan'), device=config.device)  ## CHANGED
    
    
    
    

    for epoch, (src_cpu, mask_cpu) in enumerate(
        tqdm(train_loader, total=total_epochs, desc="Epochs (1 batch each)", dynamic_ncols=True)
    ):
        src  = src_cpu.to(config.device, non_blocking=True)
        mask = mask_cpu.to(config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = get_loss(model, criterion, src, mask=mask)
        loss.backward()
        optimizer.step()
        
        

        evaluation_ready = (epoch % 1000 == 0)  ## CHANGED
        if evaluation_ready:  ## CHANGED
            
            
            #hess_lambda = compute_hessian_max_eig(
            #    model,
            #    criterion,
            #    test_set,          
            #    mask=test_mask,
            #    num_iters=80,
            #)
            
            #loss_sharpness = estimate_E_vHv_gaussian(model, criterion, test_set, mask = test_mask)
            
            #loss_sharpness = compute_sharpness_last_qkv_only(model, criterion, src_cpu, epsilon=0.1, mask=mask_cpu)
            #loss_sharpness = hessian_top_eigenvalue_last_qkv(model, criterion, src_cpu, mask=mask_cpu, num_iters=20)
 
            
            with torch.no_grad():
                model.eval()
                loss_train, train_err_1_b, train_err_2_b, _,  _,  _  = loss_err(model, criterion, src, mask)
                loss_test,  test_err_1,   test_err_2,  test_pred_batch, result_acc, last4_acc = loss_err(model, criterion, test_set, test_mask)



                #intervention based metrics
                avg_KL, _, avg_inv,_ = measure_e2_all_digits_intervention_ce_modp_extended(model, test_set, modulus)
                
                #avg_KL_shortcut, _, avg_inv_shortcut,_ = measure_e2_all_digits_intervention_ce_modp_short(model, test_short_cut, modulus)
                
                avg_KL_1, _, avg_inv_1,_ = measure_e1_abc_digits_intervention_ce_modp_extended(model, test_set, modulus)

                #partial intervention
                #avg_KL_partial, _, avg_inv_partial,_ = measure_e2_addup_intervention_ce_modp_short(model, test_set, modulus)


                
                
                
                #we first use the model to autoregressively generate 
                
                pred_autoreg = autoreg_generate_from_firstded_to_stop_1(model, test_set, p=modulus)
                


                #entropy
                _, sharpness = compute_e3_entropy_sharpness_modp_extended(model, test_set, modulus) 
                _, sharpness_pred = compute_e3_entropy_sharpness_modp_short(model, pred_autoreg, modulus) 

                
                #ratio metrics
                #four different ratio
                ratio_E2_eq_E3_eq, ratio_E2_eq_E3_neq, ratio_E2_neq_E3_eq, ratio_E2_neq_E3_neq = proportion_E2_E3_joint_cases_extended(pred_autoreg,test_set,modulus)

                

                
            
            
            record_loss  = [loss_train.item(), loss_test.item()]  ## CHANGED
            record_error = [test_err_1.item(), test_err_2.item(),
                            train_err_1_b.item(), train_err_2_b.item(), result_acc.item()]  ## CHANGED
            test_pred = test_pred_batch  ## CHANGED

            

            
            if run is not None:  ## CHANGED
                #run.log({"loss": loss_test, "error_sentence_level": test_err_1, "error_element_level": test_err_2, "Model_Prediction_Accuracy": result_acc, "E2_Intervention_KL_Divergence": avg_KL, "E1_Intervention_KL_Divergence": avg_KL_1, "MAP_Consistency_under_E2_intervention": avg_inv, "MAP_Consistency_under_E1_intervention":avg_inv_1, "Exponential_Entropy_of_Model’s_Predicted_Distribution": sharpness, "Accuracy_of_Predictions_Matching_E1":e1_rate, "Accuracy_of_Predictions_Matching_E2":e2_rate, "KL_E1":KL_E1, "KL_E2":KL_E2, "KL_Mixed":KL_Mixed}, step=epoch)
                
                run.log({"loss": loss_test, "error_sentence_level": test_err_1, "error_element_level": test_err_2, "Model_Prediction_Accuracy": result_acc, "Entropy_of_Model’s_Predicted_Distribution": sharpness,"Entropy_of_Model’s_Predicted_Distribution_E2_pred": sharpness_pred, "INR_under_E2_intervention": avg_inv, "E2_IDS": avg_KL, "INR_under_E1_intervention": avg_inv_1, "E1_IDS": avg_KL_1, "ratio_E2e_E3n": ratio_E2_eq_E3_neq, "ratio_E2n_E3e": ratio_E2_neq_E3_eq,"ratio_E2e_E3e": ratio_E2_eq_E3_eq, "ratio_E2n_E3n": ratio_E2_neq_E3_neq}, step=epoch)
                
                # "Indicator":indicator_metric, "self-reflection_head1":lambda_max, "self-reflection_head2": lambda_max_head2, "nuclear_norm_head1":nuclear_norm, "nuclear_norm_head2":nuclear_norm_head2,
                
                
                #, "unfaithfulness_ratio": unfaithfulness_ratio
                #, "avg_KL_E2": avg_KL, "avg_inv": avg_inv

        training_info.add_epoch_data(
                epoch=epoch,
                loss=record_loss,
                test_error=record_error,
                train_error=[],
                matching_scores=0,
                batch={"batch_size": config.batch_size, "learning_rate": optimizer.param_groups[0]['lr']},
                induction_score=[epoch, 0],
                simplification_score=[epoch, 0])

        scheduler.step()

        if ((epoch % getattr(config, "measurements_every_epoch", 1) == 0) or
           (epoch <= getattr(config, "measurements_initial_few_epoch", 0))):
            if getattr(config, "print_output", False):  ## CHANGED
                val1 = training_info.losses[epoch][1]
                val2 = training_info.test_errors[epoch][0]
                print(f"----> Epoch:{epoch + 1:>5}, Test Loss: {val1:.3f}, Test Error: {val2:.3f}")

            if getattr(config, "print_diagnosis", False) and test_pred is not None:  ## CHANGED
                i = min(3, test_set.size(0) - 1)
                print(decode_expressions(test_set[i].detach().cpu().unsqueeze(0), modulus=modulus, token_map=token_map)[0])  ## CHANGED
                idx = test_set_info["solution_start_ind"][i] - 1
                print(decode_expressions(test_pred[i].detach().cpu()[idx:].unsqueeze(0), modulus=modulus, token_map=token_map)[0])  ## CHANGED
                print(result_acc.item())
                print(last4_acc.item())

                
            
        #n_save = max(1, getattr(config, "n_save", 1))
        #save_every = max(1, total_epochs // n_save)
        #if (epoch + 1) % save_every == 0:
        #    with torch.no_grad():
        #        model.eval()
        #        save_checkpoints(
        #            config, epoch, test_set, model, optimizer, scheduler,
        #            record_loss, record_error, []
        #        )
        #model.train()
        
        

    return model, training_info








































def save_results(
    out_dir: str | Path,
    filename: str,
    proportion_arr: np.ndarray,
    E1_intervention_arr: np.ndarray,
    E2_intervention_arr: np.ndarray,
    E1modp_arr: np.ndarray,
    E2modp_arr: np.ndarray,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    save_path = out_dir / filename
    np.savez_compressed(
        save_path,
        proportion=proportion_arr,
        E1_intervention=E1_intervention_arr,
        E2_intervention=E2_intervention_arr,
        E1modp=E1modp_arr,
        E2modp=E2modp_arr,
    )
    return save_path