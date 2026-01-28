import json
import os
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
import torch
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.manifold import TSNE
#load checkpoints & estops
def load_checkpoints(checkpoint_dir):

    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )


    if not checkpoint_files:
        print("No checkpoints found")
        return []

    checkpoints = []
    for filename in checkpoint_files:
        file_path = os.path.join(checkpoint_dir, filename)
        checkpoint = torch.load(file_path, weights_only=False, map_location=torch.device("cpu"))
        checkpoints.append(checkpoint)
    return checkpoints


def create_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng



def fix_random_seed_with_shuffle(seed, reproduce=False):
    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.use_deterministic_algorithms(True)
     

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  

    generator = torch.Generator()
    generator.manual_seed(seed)  

    return generator



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def plot_curve(
        config,
        training_info,
        save_dir=None,
        log_x=False,
        dpi=None,
        figure_height=None,
        figure_width=None,
        vline_sent_x: float | None = None,  
        vline_token_x: float | None = None,
        is_loaded_data=False,
        ma_window: int = 5
):
   
    assert ma_window >= 1, "ma_window must >= 1"


    if is_loaded_data:
        loss_info = np.array(training_info['training_info'].losses)
        error_info = np.array(training_info['training_info'].test_errors)
    else:
        loss_info = np.array(training_info.losses)
        error_info = np.array(training_info.test_errors)


    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path_1 = "output/figures/loss-epoch.pdf"
        save_path_2 = "output/figures/error-epoch.pdf"
        save_path_3 = "output/figures/error-combined.pdf"
    else:
        save_path_1 = os.path.join(save_dir, "loss-epoch.pdf")
        save_path_2 = os.path.join(save_dir, "error-epoch.pdf")
        save_path_3 = os.path.join(save_dir, "error-combined.pdf")

    #-------------------------Loss----------------------------
    num_epoch, num_elements = loss_info.shape
    fig, axs = plt.subplots(1, num_elements,
                            figsize=(figure_width * num_elements, figure_height),
                            dpi=dpi)
    titles_loss = ["Train Loss", "Test Loss"] if not config.curriculum else \
        ["Train Loss", "Normal Loss", "Curr Loss", "Simplest Loss"]

    for i in range(num_elements):
        series = loss_info[:, i]
        if ma_window > 1:
            smoothed = np.array([
                series[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
        else:
            smoothed = series

        axs[i].plot(smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
        axs[i].set_xlabel("Epochs", weight="bold")
        axs[i].set_ylabel("Loss", weight="bold")
        axs[i].set_title(titles_loss[i], weight="bold")
        axs[i].grid(True)
        if log_x:
            axs[i].set_xscale("log")
            axs[i].xaxis.set_major_locator(LogLocator(base=10.0))

        
        if vline_sent_x is not None:
            axs[i].axvline(x=vline_sent_x, color='red', linestyle='--', linewidth=1)
        if vline_token_x is not None:
            axs[i].axvline(x=vline_token_x, color='red', linestyle='--', linewidth=1)
        

    plt.savefig(save_path_1, bbox_inches="tight")
    plt.close(fig)

    #-------------------------Error----------------------------
    num_epoch, num_elements = error_info.shape
    fig2, axs2 = plt.subplots(1, num_elements,
                              figsize=(figure_width * num_elements, figure_height),
                              dpi=dpi)
    titles_error = ["Test Error (Sentence Level) ", "Test Error (Token Level)",
                    "Train Error (Sentence Level) ", "Train Token (Token Level)"]

    for i in range(num_elements):
        series = error_info[:, i]
        if ma_window > 1:
            smoothed = np.array([
                series[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
        else:
            smoothed = series

        axs2[i].plot(smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
        axs2[i].set_xlabel("Epochs", weight="bold")
        axs2[i].set_ylabel("Error", weight="bold")
        axs2[i].set_title(titles_error[i], weight="bold")
        axs2[i].grid(True)
        if log_x:
            axs2[i].set_xscale("log")
            axs2[i].xaxis.set_major_locator(LogLocator(base=10.0))

        # Draw vertical lines
        if i in (0,2) and vline_sent_x is not None:
            axs2[i].axvline(x=vline_sent_x, color='red', linestyle='--', linewidth=1)
        if i in (1,3) and vline_token_x is not None:
            axs2[i].axvline(x=vline_token_x, color='red', linestyle='--', linewidth=1)

    plt.savefig(save_path_2, bbox_inches="tight")
    plt.close(fig2)
    
     # ------------------------- Combined Error ----------------------------
    fig3, axs3 = plt.subplots(1, 2, figsize=(figure_width * 2, figure_height), dpi=dpi)
    test_sent = error_info[:, 0]
    test_tok  = error_info[:, 1]
    train_sent = error_info[:, 2]
    train_tok  = error_info[:, 3]

    for ax, series_sent, series_tok, phase in zip(
            axs3,
            [test_sent, train_sent],
            [test_tok, train_tok],
            ["Test", "Train"]
    ):
        if ma_window > 1:
            smoothed_sent = np.array([
                series_sent[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
            smoothed_tok = np.array([
                series_tok[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
        else:
            smoothed_sent, smoothed_tok = series_sent, series_tok

        ax.plot(smoothed_sent, marker='o', linestyle='-', linewidth=1, markersize=3, label=f"{phase} Sent Err")
        ax.plot(smoothed_tok, marker='x', linestyle='-', linewidth=1, markersize=3, label=f"{phase} Token Err")
        ax.set_xlabel("Epochs", weight="bold")
        ax.set_ylabel("Error", weight="bold")
        ax.set_title(f"{phase} Error Levels", weight="bold")
        ax.grid(True)
        ax.legend()

        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0))

        # Vertical lines on combined plots
        if phase == "Test":
            if vline_sent_x is not None:
                ax.axvline(x=vline_sent_x, color='red', linestyle='--', linewidth=1)
            if vline_token_x is not None:
                ax.axvline(x=vline_token_x, color='red', linestyle='--', linewidth=1)
        else:  # Train
            if vline_sent_x is not None:
                ax.axvline(x=vline_sent_x, color='red', linestyle='--', linewidth=1)
            if vline_token_x is not None:
                ax.axvline(x=vline_token_x, color='red', linestyle='--', linewidth=1)

    plt.savefig(save_path_3, bbox_inches="tight")
    plt.close(fig3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


def plot_training_curve(
    training_info,
    save_dir=None,
    log_x=False,
    dpi=None,
    figure_height=4,
    figure_width=6,
    vline_x: float | None = None,
    is_loaded_data=False,
    ma_window: int = 1
):
    
    assert ma_window >= 1, "ma_window must >= 1"


    if is_loaded_data:
        train_error_info = np.array(training_info['training_info'].train_errors)
    else:
        train_error_info = np.array(training_info.train_errors)

    labels = np.unique(train_error_info[:, 0])
    num_epoch = train_error_info.shape[0]


    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path = os.path.join("output/figures", "training_error.pdf")
    else:
        save_path = os.path.join(save_dir, "training_error.pdf")


    cmap = plt.get_cmap("viridis", len(labels))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)
    for idx, label in enumerate(labels):
        subset = train_error_info[train_error_info[:, 0] == label]
        epochs = subset[:, 1].astype(int)
        errors = subset[:, 2].astype(float)

    
        if ma_window > 1:
            smoothed = np.array([
                errors[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(len(errors))
            ])
        else:
            smoothed = errors

        ax.plot(
            epochs,
            smoothed,
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=3,
            label=f"Label {int(label)}",
            color=cmap(idx)
        )

    ax.set_title("Training Error over Epochs", weight="bold")
    ax.set_xlabel("Epoch", weight="bold")
    ax.set_ylabel("Error", weight="bold")
    ax.grid(True)
    ax.legend()

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))


    if vline_x is not None:
        ax.axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



















def matching_scores(config, model, r=10):
    num_layers = config.num_layers+1
    d_model = config.d_model
    matching_scores = []

    model_list = model.h_1 + [model.h_2]

    for i in range((num_layers-1)):

        W_q = model_list[i+1].mha.W_q.weight
        W_k = model_list[i+1].mha.W_k.weight
        W_v = model_list[i].mha.W_v.weight
        W_o = model_list[i].mha.W_o.weight

        W_QK = (W_q.T @ W_k)/torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        W_OV = W_o @ W_v

        U_QK, S_QK, Vh_QK = torch.linalg.svd(W_QK)
        U_OV, S_OV, Vh_OV = torch.linalg.svd(W_OV)

        sim_inner = torch.max(torch.linalg.svdvals(Vh_QK[:(r+1), :] @ U_OV[:, :(r+1)]))
        sim_outer = torch.max(torch.linalg.svdvals(Vh_OV[:(r+1), :] @ U_QK[:, :(r+1)]))

        matching_scores = matching_scores + [sim_inner.item(), sim_outer.item()]

    return matching_scores




def plot_qk_subspace_matching(model, fig_name, config):
    num_svals_plot = 128
    num_layers = config.num_layers+1
    d_model = config.d_model
    model_list = model.h_1 + [model.h_2]

    s_match = torch.zeros(2, (num_layers-1), num_svals_plot)

    for j in range(num_svals_plot):
        for i in range((num_layers-1)):
            W_q = model_list[i + 1].mha.W_q.weight
            W_k = model_list[i + 1].mha.W_k.weight
            W_v = model_list[i].mha.W_v.weight
            W_o = model_list[i].mha.W_o.weight

            W_QK = (W_q.T @ W_k) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
            W_OV = W_o @ W_v

            U_QK, S_QK, Vh_QK = torch.linalg.svd(W_QK)
            U_OV, S_OV, Vh_OV = torch.linalg.svd(W_OV)

            _, s, _ = torch.linalg.svd(Vh_QK[:(j + 1), :] @ U_OV[:, :(j + 1)])
            _, s2, _ = torch.linalg.svd(Vh_OV[:(j + 1), :] @ U_QK[:, :(j + 1)])

            s_match[0,i,j] = s[0]
            s_match[1,i,j] = s2[0]


    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    axs[0].plot(S_QK[:num_svals_plot] / S_QK[0], "-o", label="qk", linewidth=2)
    axs[0].plot(S_OV[:num_svals_plot] / S_OV[0], "-o", label="ov", linewidth=2)
    axs[0].plot(s_match[0, :num_svals_plot], "-o", label="match")
    axs[0].legend()
    axs[0].set_title("inner match")
    axs[1].plot(S_QK[:num_svals_plot] / S_QK[0], "-o", label="qk", linewidth=2)
    axs[1].plot(S_OV[:num_svals_plot] / S_OV[0], "-o", label="ov", linewidth=2)
    axs[1].plot(s_match[1, :num_svals_plot], "-o", label="match")
    axs[1].legend()
    axs[1].set_title("outer match")

    plt.savefig(fig_name)
    plt.close()






def plot_subspace_matching(
    training_info,
    save_dir=None,
    log_x=False,
    dpi=None,
    figure_height=None,
    figure_width=None,
    vline_x: float | None = None,
    is_loaded_data=False,
    ma_window: int = 1
):
    
    assert ma_window >= 1, "ma_window must >= 1"


    if is_loaded_data:
        matching_scores = np.array(training_info['training_info'].matching_scores)
    else:
        matching_scores = np.array(training_info.matching_scores)

    num_epoch, total = matching_scores.shape
    num_layers = total // 2

  
    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path_inner = "output/figures/inner-matching_scores.pdf"
        save_path_outer = "output/figures/outer-matching_scores.pdf"
    else:
        save_path_inner = os.path.join(save_dir, "inner-matching_scores.pdf")
        save_path_outer = os.path.join(save_dir, "outer-matching_scores.pdf")

    #---------------------Inner matching----------------------
    fig, axs = plt.subplots(1, num_layers,
                            figsize=(figure_width * num_layers, figure_height),
                            dpi=dpi)
    for i in range(num_layers):
        series = matching_scores[:, 2*i]
  
        if ma_window > 1:
            smoothed = np.array([
                series[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
        else:
            smoothed = series

        axs[i].plot(smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
        axs[i].set_xlabel("Epochs", weight="bold")
        axs[i].set_ylabel("Inner Matching Score", weight="bold")
        axs[i].set_title(f"Layer {i+1}", weight="bold")
        axs[i].grid(True)
        if log_x:
            axs[i].set_xscale("log")
            axs[i].xaxis.set_major_locator(LogLocator(base=10.0))

        if vline_x is not None:
            axs[i].axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.savefig(save_path_inner, bbox_inches="tight")
    plt.close(fig)

    #----------------------Outer matching-------------------------
    fig2, axs2 = plt.subplots(1, num_layers,
                              figsize=(figure_width * num_layers, figure_height),
                              dpi=dpi)
    for i in range(num_layers):
        series = matching_scores[:, 2*i+1]
        if ma_window > 1:
            smoothed = np.array([
                series[max(0, j - ma_window + 1): j + 1].mean()
                for j in range(num_epoch)
            ])
        else:
            smoothed = series

        axs2[i].plot(smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
        axs2[i].set_xlabel("Epochs", weight="bold")
        axs2[i].set_ylabel("Outer Matching Score", weight="bold")
        axs2[i].set_title(f"Layer {i+1}", weight="bold")
        axs2[i].grid(True)
        if log_x:
            axs2[i].set_xscale("log")
            axs2[i].xaxis.set_major_locator(LogLocator(base=10.0))
        if vline_x is not None:
            axs2[i].axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.savefig(save_path_outer, bbox_inches="tight")
    plt.close(fig2)




def plot_elbow_qkov(model, config, fig_name, layer_index=1, save_dir=None, show_elbow=True):
    num_svals_plot = config.d_model
    d_model = config.d_model

    model_list = model.h_1 + [model.h_2]
    s_match = torch.zeros(2, num_svals_plot)

    if save_dir is None:
        if not os.path.isdir("output/figures"):
            os.makedirs("output/figures", exist_ok=True)
        save_path = os.path.join("output/figures", fig_name)
    else:
        save_path = os.path.join(save_dir, fig_name)

    with torch.no_grad():
        W_q = model_list[layer_index].mha.W_q.weight
        W_k = model_list[layer_index].mha.W_k.weight
        W_v = model_list[layer_index - 1].mha.W_v.weight
        W_o = model_list[layer_index - 1].mha.W_o.weight

        W_QK = (W_q.T @ W_k) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        W_OV = W_o @ W_v

        U_QK, S_QK, Vh_QK = torch.linalg.svd(W_QK)
        U_OV, S_OV, Vh_OV = torch.linalg.svd(W_OV)

        S_QK = S_QK.cpu().numpy()
        S_OV = S_OV.cpu().numpy()

        for j in range(num_svals_plot):
            _, s, _ = torch.linalg.svd(Vh_QK[:(j + 1), :] @ U_OV[:, :(j + 1)])
            _, s2, _ = torch.linalg.svd(Vh_OV[:(j + 1), :] @ U_QK[:, :(j + 1)])

            s_match[0, j] = s[0]
            s_match[1, j] = s2[0]

    match_inner = s_match[0, :num_svals_plot].cpu().numpy()
    match_outer = s_match[1, :num_svals_plot].cpu().numpy()

    #--------------- Choose the best rank of the matrix ------------------
    def find_elbow(spectrum):
        diffs = np.diff(spectrum)
        second_diff = np.diff(diffs)
        elbow_index = np.argmax(second_diff) + 2  # +2因为二阶差分少两个长度
        return elbow_index

    elbow_inner = find_elbow(match_inner)
    elbow_outer = find_elbow(match_outer)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(S_QK[:num_svals_plot] / S_QK[0], "-o", label="qk spectrum", linewidth=2)
    axs[0].plot(S_OV[:num_svals_plot] / S_OV[0], "-o", label="ov spectrum", linewidth=2)
    axs[0].plot(match_inner, "-o", label="inner match", color='green')
    if show_elbow:
        axs[0].axvline(elbow_inner, color='red', linestyle='--', label=f"elbow r={elbow_inner}")
    axs[0].legend()
    axs[0].set_title(f"Inner Match (Layer {layer_index})")

    axs[1].plot(S_QK[:num_svals_plot] / S_QK[0], "-o", label="qk spectrum", linewidth=2)
    axs[1].plot(S_OV[:num_svals_plot] / S_OV[0], "-o", label="ov spectrum", linewidth=2)
    axs[1].plot(match_outer, "-o", label="outer match", color='orange')
    if show_elbow:
        axs[1].axvline(elbow_outer, color='red', linestyle='--', label=f"elbow r={elbow_outer}")
    axs[1].legend()
    axs[1].set_title(f"Outer Match (Layer {layer_index})")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return elbow_inner, elbow_outer


def extract_hidden(model, seq, config):
    """
    extract hidden states from all sublayers
    """
    L, H = config.num_layers + 1, config.num_heads  # add one to L
    D = config.d_model
    B, T = seq.size(0), seq.size(1)  # batch_size, seq_len
    hiddens = torch.zeros(L, 4, B, T, D)

    pad_id = config.pad_id
    pad_mask = (seq != pad_id) 


    causal_mask = torch.tril(
        torch.ones((T, T), dtype=torch.bool, device=config.device)
    )


    mask = causal_mask.unsqueeze(0).unsqueeze(0) & \
                pad_mask.unsqueeze(1).unsqueeze(2)

    block_list = [model.h_1[j] for j in range(L - 1)] + [model.h_2]  # awkward notation
    x = model.embed(seq)
    with torch.no_grad():
        for layer, block in enumerate(block_list):
            attn_output, _ = block.mha(x, x, x, mask, output_attn=True)
            hiddens[layer, 0] = attn_output
            x2 = x + attn_output
            hiddens[layer, 1] = x2
            x3 = block.feed_forward(x2)
            hiddens[layer, 2] = x3
            hiddens[layer, 3] = x2 + x3
            x, _ = block(x, mask)
    return hiddens


def prediction_heatmaps(
        model,
        seq: torch.LongTensor,        # (T,)
        config,
        task,                         
        position: int,                
        dpi: int = 200,
        save_path: Optional[str] = None
):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch

    device = next(model.parameters()).device
    seq = seq.to(device)
    T = seq.size(0)

    if not (0 <= position < T):
        raise ValueError(f"position {position} out of range [0, {T})")


    ids_np     = seq.cpu().numpy()
    sample_str = task.map_ids_to_str(ids_np)

    number_to_str = task.map  # id->str 映射
    target_idx     = position

    # 1) base distribution
    with torch.no_grad():
        logits_all, _ = model(seq.unsqueeze(0))      # (1, T, V)
    base_probs = torch.softmax(logits_all[0, target_idx], dim=-1)  # (V,)


    hiddens = extract_hidden(model, seq.unsqueeze(0), config).cpu().numpy()
    L, _, _, _, _ = hiddens.shape


    fig, axes = plt.subplots(1, L, figsize=(4 * L, 6), dpi=dpi, squeeze=False)
    axes = axes[0]
    fig.suptitle(sample_str, fontsize=16, weight="bold", y=1.02)

    comp_names = ["sa", "res+sa", "mlp", "res+mlp"]
    for layer in range(L):
        ax = axes[layer]
        if layer < L - 1:
            comps = ["base", "sa", "res+sa"]
        else:
            comps = ["base", "sa", "res+sa", "mlp", "res+mlp"]

        rows = []
        for name in comps:
            if name == "base":
                rows.append(base_probs.detach().cpu().numpy())
            else:
                ci = comp_names.index(name)
                h_vec = hiddens[layer, ci, 0, target_idx]       # (D,)
                h_ten = torch.from_numpy(h_vec).to(device)
                logits = model.fc(h_ten)                       # (V,)
                probs  = torch.softmax(logits, dim=-1)
                rows.append(probs.detach().cpu().numpy())

        mat = np.stack(rows, axis=0)  # (len(comps), V)
        
        mat = mat[:, :15]
        xticks = number_to_str[:15]

        sns.heatmap(
            mat, ax=ax,
            xticklabels=xticks,
            yticklabels=comps,
            vmin=0, vmax=1,
            linewidths=0.1, linecolor="white",
            cbar=(layer == L - 1)
        )
        ax.set_title(f"Layer {layer+1}", weight="bold")
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

    plt.tight_layout(rect=[0,0,1,0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()




def autoregressive_extend(model, input_ids: torch.LongTensor, target_len: int) -> torch.LongTensor:
    
    model.eval()
    generated = input_ids.clone()  # (B, T_curr)

    with torch.no_grad():
        while generated.size(1) < target_len:
         
            logits, _ = model(generated)  # (B, T_curr, V)
            next_id = logits[:, -1, :].argmax(dim=-1)  # (B,)
            next_id = next_id.unsqueeze(1)  # (B,1)

       
            generated = torch.cat([generated, next_id], dim=1)  # (B, T_curr+1)

    return generated


def find_reduction_indices(ids: np.ndarray, tokenizer) -> Tuple[int,int]:
   
    _VOCAB = tokenizer["vocab"]
    open_id, close_id = _VOCAB["parentheses"]


    close_positions = [i for i,x in enumerate(ids) if x == close_id]
    if not close_positions:
       
        return 0, 2

   
    id2 = min(close_positions)
    
    id1 = max(i for i in range(id2) if ids[i] == open_id)

    dist = id2 - id1
    if dist == 2:
        
        return id1+1, id1+1
    if dist == 4:
        # "( v op v )"
        return id1+1, id1+3
    
    return id1+1, id1+3


def augment_congruence(ids: np.ndarray, shift: int, tokenizer) -> np.ndarray:

    new_ids = ids.copy()
    try:
        i1, i2 = find_reduction_indices(ids, tokenizer)
    except Exception:
        return new_ids

  
    new_ids[i1] = (int(new_ids[i1]) + shift) % 10
    new_ids[i2] = (int(new_ids[i2]) + shift) % 10
    return new_ids


def induction_score(
        model: torch.nn.Module,
        seqs: torch.LongTensor,  
        task,  
        config  
        
) -> float:

    device = next(model.parameters()).device
    seqs = seqs.to(device)
    B, T = seqs.size()

 
    seq_ids_batch = seqs.detach().cpu().numpy()
    all_pairs: List[List[Tuple[int, int]]] = [task.pair_copy_tokens(seq_ids) for seq_ids in seq_ids_batch]

   
    with torch.no_grad():
        _, attn_mats = model(seqs)
    attn_probs, _ = attn_mats[2]  
    attn_avg = attn_probs[:, 0]  
   
    scores = []
    for b in range(B):
        pairs = all_pairs[b]
        if len(pairs) == 0:
            continue  
        attn = attn_avg[b]  # (T, T)
        vals = [attn[tgt, src + 1].item() for src, tgt in pairs]
        scores.append(sum(vals) / len(vals))

  
    if len(scores) == 0:
        return float('nan') 
    return sum(scores) / len(scores)


def plot_induction_score(
        training_info,
        save_dir=None,
        log_x=False,
        dpi=200,
        figure_height=4,
        figure_width=6,
        vline_x: float | None = None,
        is_loaded_data=False,
        ma_window: int = 1
        
):
   
    assert ma_window >= 1, "ma_window must >= 1"

    
    if is_loaded_data:
        scores = np.array(training_info['training_info'].induction_scores)
    else:
        scores = np.array(training_info.induction_scores)

    epochs = scores[:, 0]
    values = scores[:, 1]
    num_epoch = len(epochs)

  
    if ma_window > 1:
        smoothed = np.array([
            values[max(0, j - ma_window + 1): j + 1].mean()
            for j in range(num_epoch)
        ])
    else:
        smoothed = values

    
    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path = "output/figures/induction_scores.pdf"
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "induction_scores.pdf")


    plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    plt.plot(epochs, smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.xlabel("Epoch", weight="bold")
    plt.ylabel("Induction Score", weight="bold")
    plt.title("Induction Score Over Training", weight="bold")
    plt.grid(True)

    if log_x:
        plt.xscale("log")
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))

    if vline_x is not None:
        plt.axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def simplification_head_score(
        model: torch.nn.Module,
        seqs: torch.LongTensor,  # (B, T)
        task,  
        tokenizer,
        config  
) -> float:
    
    device = next(model.parameters()).device
    seqs = seqs.to(device)
    B, T = seqs.size()

    
    seq_ids_batch = seqs.detach().cpu().numpy()
    all_result_ind: List[int] = [task.find_one_step_result_index(seq_ids) for seq_ids in seq_ids_batch]
    all_number_ind = [find_reduction_indices(seq_ids, tokenizer) for seq_ids in seq_ids_batch]
    num_ind_1, num_ind_2 = zip(*all_number_ind)
    
    with torch.no_grad():
        _, attn_mats = model(seqs)
    attn_probs, _ = attn_mats[2]  
    attn_avg = attn_probs[:, 0] 


    scores = []
    for b in range(B):
        result_ind = all_result_ind[b]
        ind_1 = num_ind_1[b]
        ind_2 = num_ind_2[b]
        attn = attn_avg[b]
        scores.append(attn[result_ind-1,ind_1-1:ind_2+2].sum().item())

    if len(scores) == 0:
        return float('nan') 
    return sum(scores) / len(scores)




def plot_simplification_score(
        training_info,
        save_dir=None,
        log_x=False,
        dpi=200,
        figure_height=4,
        figure_width=6,
        vline_x: float | None = None,
        is_loaded_data=False,
        ma_window: int = 1
):

    assert ma_window >= 1, "ma_window must >= 1"

 
    if is_loaded_data:
        scores = np.array(training_info['training_info'].simplification_scores)
    else:
        scores = np.array(training_info.simplification_scores)

    epochs = scores[:, 0]
    values = scores[:, 1]
    num_epoch = len(epochs)

  
    if ma_window > 1:
        smoothed = np.array([
            values[max(0, j - ma_window + 1): j + 1].mean()
            for j in range(num_epoch)
        ])
    else:
        smoothed = values


    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path = "output/figures/simplification_scores.pdf"
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "simplification_scores.pdf")


    plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    plt.plot(epochs, smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.xlabel("Epoch", weight="bold")
    plt.ylabel("Simplification Score", weight="bold")
    plt.title("Simplification Score Over Training", weight="bold")
    plt.grid(True)

    if log_x:
        plt.xscale("log")
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))

    if vline_x is not None:
        plt.axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()







def eval_opposite_structure(
    test_ids: np.ndarray | torch.Tensor,
    model: torch.nn.Module,
    tsne_perplexity: int = 30,
    tsne_n_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
    device = next(model.parameters()).device
    model.eval()


    if isinstance(test_ids, np.ndarray):
        ids = torch.from_numpy(test_ids).long().to(device)
    else:
        ids = test_ids.long().to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(1)
    B, L = ids.shape

    # 2) embedding + positional
    if model.pos in ["relative", "rotary"]:
        x = model.embed(ids)
    else:
        x = model.pos_embed(model.embed(ids))

    mask = torch.tril(torch.ones(L, L, device=device)).view(1,1,L,L)
    out = x
    for block in model.h_1:
        out, _ = block(out, mask)


    last = model.h_2
    sa_in  = last.ln_1(out) if last.norm else out
    sa_out = last.mha(sa_in, sa_in, sa_in, mask, output_attn=False)
    sa_out = last.dropout1(sa_out) if last.drop is not None else sa_out
    res_out = out + sa_out if last.residual else sa_out


    pre_ffn = last.ln_2(res_out) if last.norm else res_out   # [B,L,d_model]
    vecs    = pre_ffn.reshape(-1, pre_ffn.size(-1))          # [B*L, d_model]


    labels_all = ids.reshape(-1)                             # [B*L]
    valid_mask = (labels_all >= 0) & (labels_all < 10)
    vecs        = vecs[valid_mask]                           # [M, d_model]
    labels      = labels_all[valid_mask].cpu().numpy()       # [M,]


    vecs_np = vecs.detach().cpu().numpy()
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        init='random',
        learning_rate='auto',
        verbose=1
    )
    X3 = tsne.fit_transform(vecs_np)                         # [M,3]

 
    centroids = np.zeros((10, 2), dtype=float)
    for k in range(10):
        pts = X3[labels == k]
        centroids[k] = pts.mean(axis=0)


    pairs = [(1,9), (2,8), (3,7), (4,6)]
    opposite_angles = []
    for i, j in pairs:
        v1, v2 = centroids[i], centroids[j]
        cos_ = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_ = np.clip(cos_, -1.0, 1.0)
        angle = np.arccos(cos_)
        opposite_angles.append(angle)
    opposite_angles = np.array(opposite_angles)  # [4,]

   
    deviations = np.abs(opposite_angles - np.pi) / np.pi
    opposite_score = float((1.0 - deviations).mean())

    return opposite_score







def plot_structure_score(
        training_info,
        save_dir=None,
        log_x=False,
        dpi=200,
        figure_height=4,
        figure_width=6,
        vline_x: float | None = None,
        is_loaded_data=False,
        ma_window: int = 1
):

    assert ma_window >= 1, "ma_window must >= 1"


    if is_loaded_data:
        scores = np.array(training_info['training_info'].structure_scores)
    else:
        scores = np.array(training_info.structure_scores)

    epochs = scores[:, 0]
    values = scores[:, 1]
    num_epoch = len(epochs)


    if ma_window > 1:
        smoothed = np.array([
            values[max(0, j - ma_window + 1): j + 1].mean()
            for j in range(num_epoch)
        ])
    else:
        smoothed = values

  
    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path = "output/figures/structure_scores.pdf"
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "structure_scores.pdf")

  
    plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    plt.plot(epochs, smoothed, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.xlabel("Epoch", weight="bold")
    plt.ylabel("Opposite‐pair Score", weight="bold")
    plt.title("Opposite‐pair Score Over Training", weight="bold")
    plt.grid(True)

    if log_x:
        plt.xscale("log")
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))

    if vline_x is not None:
        plt.axvline(x=vline_x, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def plot_and_save_subspace_matching(
    model: torch.nn.Module,
    config,
    layer_pairs=[(0, 1), (0, 2), (1, 2)],
    r: int = 10,
    method: str = "top",
    save_dir = None,
    dpi: int = 300
):
   
    H = config.num_heads

    def extract_circuits(block):
        d_model = config.d_model
        d_head = d_model // H
        W_q = block.mha.W_q.weight.view(H, d_head, d_model)
        W_k = block.mha.W_k.weight.view(H, d_head, d_model)
        W_v = block.mha.W_v.weight.view(H, d_head, d_model)
        W_o = block.mha.W_o.weight.view(d_model, H, d_head)
        W_qk = torch.zeros(H, d_model, d_model)
        W_ov = torch.zeros(H, d_model, d_model)
        with torch.no_grad():
            for h in range(H):
                W_qk[h] = W_q[h].T @ W_k[h]
                W_ov[h] = W_o[:,h] @ W_v[h]
        return W_qk.cpu(), W_ov.cpu()


    def subspace_matching(block1, block2):
        W_qk1, W_ov1 = extract_circuits(block1)
        W_qk2, W_ov2 = extract_circuits(block2)
 
        write_list = [W_ov1[h] for h in range(H)] + [block1.feed_forward.fc2.weight.cpu()]
 
        read_list  = [W_qk2[h] for h in range(H)] + [block2.feed_forward.fc1.weight.cpu()]
        scores = torch.zeros(len(write_list), len(read_list))
        for i, Ww in enumerate(write_list):
            for j, Wr in enumerate(read_list):
                U1, _, _ = torch.linalg.svd(Ww)
                _, _, Vt2 = torch.linalg.svd(Wr)
                # principal angles between span(U1[:,:r]) and span(Vt2[:r,:])
                M = Vt2[:r] @ U1[:, :r]
                U3, S3, Vt3 = torch.linalg.svd(M)
                if method == "top":
                    scores[i, j] = S3[0]
                else:
                    scores[i, j] = torch.sqrt((S3 ** 2).mean())
        return scores.detach().numpy()


    blocks = list(model.h_1) + [model.h_2]

    scores_all = []
    for (i, j) in layer_pairs:
        scores_all.append(subspace_matching(blocks[i], blocks[j]))

 
    if H == 1:
        labels = ['sa', 'FFN']
    else:
        labels = [f'Head {h}' for h in range(H)] + ['FFN']

    if save_dir is None:
        os.makedirs("output/figures", exist_ok=True)
        save_path = "output/figures/layer_matching_heatmap.pdf"
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "layer_matching_heatmap.pdf")


    fig, axs = plt.subplots(1, len(layer_pairs), figsize=(6*len(layer_pairs), 5))
    for ax, scores, (i, j) in zip(axs, scores_all, layer_pairs):
        sns.heatmap(
            scores, ax=ax, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'similarity'}
        )
        ax.set_title(f"Layer {i} vs {j}", weight="bold")
        ax.set_xlabel("read-circuit", weight="bold")
        ax.set_ylabel("write-circuit", weight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)










class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


        
        

from dataclasses import dataclass
from typing import List, Dict





@dataclass
class TrainingInfo:
    epochs: List[int] = field(default_factory=list)  # List of epoch indices
    losses: List[List[float]] = field(default_factory=list)  # List of loss lists
    test_errors: List[List[float]] = field(default_factory=list)  # List of error lists
    batchwise_train_errors: List[List[float]] = field(default_factory=list)
    train_errors: List[List[float]] = field(default_factory=list)
    matching_scores: List[List[float]] = field(default_factory=list)
    batch_info: List[Dict[str, float]] = field(default_factory=list)
    induction_scores: List[List[float]] = field(default_factory=list)
    simplification_scores: List[List[float]] = field(default_factory=list)


    def add_epoch_data(self,
                       epoch: int,
                       loss: List[float],
                       test_error: List[float],
                       train_error: List[float],
                       matching_scores: List[float],
                       batch: Dict[str, float],
                       induction_score: List[float],
                       simplification_score: List[float]
                       ):
        """
        Add data for a single epoch.
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.test_errors.append(test_error)
        self.train_errors.append(train_error)
        self.matching_scores.append(matching_scores)
        self.batch_info.append(batch)
        self.induction_scores.append(induction_score)
        self.simplification_scores.append(simplification_score)

