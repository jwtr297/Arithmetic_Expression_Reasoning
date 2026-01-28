import json
import os
import warnings  ## CHANGED
import fire
import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from model.models import TFModel
from arithmetic_experiments.minimal_task_experiment.Multiplication_addition_training_wop import make_scheduler, train_with_dataloader, save_results  ## CHANGED
from arithmetic_experiments.minimal_task_experiment.utils import create_folder, fix_random_seed, seed_worker, fix_random_seed_with_shuffle
from arithmetic_experiments.minimal_task_experiment.unfaithfulness_metrics_wop import autoreg_generate_from_firstded_to_stop, proportion_E1_eq_r_but_E2_neq_r, measure_e2_all_digits_intervention_ce_modp, e2_modp_vs_model_ratio, e1_modp_vs_model_ratio,autoreg_generate_from_firstded_to_stop_1, proportion_E1_eq_r_but_E2_neq_r_1, measure_e2_all_digits_intervention_ce_modp_short, measure_e3_agreement_with_e1_e2_modp, measure_e1_ab_digits_intervention_ce_modp_short
print(torch.__version__)

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        
        
def main(p1, p2, batch_limit: int = 62500, num_epochs: int = 10, modulus: int = 11, **kwargs):

    # ---------- helper: normalize p1/p2 into iterables ----------
    def _to_iter(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        # allow numpy/torch scalars or plain numbers
        try:
            _ = float(x)
            return [x]
        except Exception:
            # e.g., generator/np array/torch tensor
            return list(x)

    p1_grid = _to_iter(p1)
    p2_grid = _to_iter(p2)
    R, C = len(p1_grid), len(p2_grid)
    



    
    # ---------- helper: filename-safe formatting (e.g., 0.1 -> "0_1") ----------
    def fmt(v):
        s = str(v)
        s = s.replace(".", "_")
        s = s.replace("/", "_")
        s = s.replace(" ", "")
        return s

    # ---------- load and override config ----------
    with open("./config1.yaml", "r") as file:
        config_args = yaml.safe_load(file)

    for k, v in kwargs.items():
        if k not in config_args:
            print(f"Warning: {k} is not supported!")
        if v != config_args.get(k):
            print(f"{k} is overloaded from {config_args.get(k)} to {v}")
            config_args[k] = v

    config = Config(**config_args)

    # ---------- seed & out dirs ----------
    generator = fix_random_seed_with_shuffle(config.seed, reproduce=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    OUT_DIR = os.path.join(BASE_DIR, "output")

    ckpt_root = os.path.join(OUT_DIR, "MA_checkpoints")
    info_root = os.path.join(OUT_DIR, "training_info")

    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(info_root, exist_ok=True)

    # dump the resolved config once
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # ---------- sweep over p1 (folder) and p2 (runs) ----------
    for i_p1, p1_val in enumerate(p1_grid):
        # Per-p1 folder under MA_checkpoints
        p1_dir_name = f"{modulus}_{fmt(p1_val)}"  # <modulus>_<p1>
        p1_dir = os.path.join(ckpt_root, p1_dir_name)
        os.makedirs(p1_dir, exist_ok=True)

        for j_p2, p2_val in enumerate(p2_grid):
            print(f"\n=== Running training for p1={p1_val}, p2={p2_val}, modulus={modulus} ===")
            user_modulus = modulus

            # --------- load dataset ----------
            data_dir = f"./tasks/minimal_task/3steps_multiplication_addition_dataset/mod{modulus}_{p1_val:.3f}_{p2_val:.3f}.pt"
            train_blob = torch.load(data_dir, weights_only=False)

            train_info = None
            vocab = None

            # Handle dict / (seqs, vocab, info) / raw tensor
            if isinstance(train_blob, dict):
                seq_tensor = train_blob.get("sequences")
                if seq_tensor is None:
                    seq_tensor = train_blob.get("train_set")
                if seq_tensor is None:
                    raise KeyError("Dataset dictionary must contain 'sequences' or 'train_set'.")
                train_set = seq_tensor.long()
                train_info = train_blob.get("info")
                dataset_modulus = train_blob.get("modulus")
                if dataset_modulus is not None and dataset_modulus != modulus:
                    warnings.warn(f"Dataset modulus {dataset_modulus} overrides requested modulus {user_modulus}.")
                    modulus = dataset_modulus
                vocab = train_blob.get("vocab")
                if vocab is not None and hasattr(vocab, "tokens"):
                    config.pad_id = vocab.tokens.get("PAD", modulus + 5)
                    config.vocab_size = max(vocab.tokens.values()) + 1
                else:
                    config.pad_id = modulus + 5
                    config.vocab_size = modulus + 6
            else:
                # try tuple form: (seqs, vocab, info)
                parsed_ok = False
                try:
                    seqs, vocab, train_info = train_blob
                    train_set = seqs.long()
                    dataset_modulus = (train_info or {}).get("modulus")
                    if dataset_modulus is not None and dataset_modulus != modulus:
                        warnings.warn(f"Dataset modulus {dataset_modulus} overrides requested modulus {user_modulus}.")
                        modulus = dataset_modulus
                    if vocab is not None and hasattr(vocab, "tokens"):
                        config.pad_id = vocab.tokens.get("PAD", modulus + 5)
                        config.vocab_size = max(vocab.tokens.values()) + 1
                    else:
                        config.pad_id = modulus + 5
                        config.vocab_size = modulus + 6
                    parsed_ok = True
                except Exception:
                    parsed_ok = False

                if not parsed_ok:
                    # assume raw tensor
                    train_set = train_blob.long()
                    config.pad_id = modulus + 5
                    config.vocab_size = modulus + 6

            # --------- cap samples & make mask ----------
            batch_size = config.batch_size
            max_samples = min(batch_limit * batch_size, train_set.size(0))
            train_set = train_set[:max_samples].to(torch.long)
            test_size = 1000
            test_set = train_set[:test_size]
            

            seq_len = train_set.size(1)
            if hasattr(config, "max_seq_len") and seq_len < getattr(config, "max_seq_len"):
                warnings.warn(
                    f"Training sequence length {seq_len} is smaller than configured max_seq_len {config.max_seq_len}."
                )

            if train_info and "solution_start_ind" in train_info:
                start_inds = train_info["solution_start_ind"][:max_samples]
                train_mask = torch.ones((max_samples, train_set.size(1)), dtype=torch.int64)
                for i in range(max_samples):
                    start = max(0, int(start_inds[i]) - 1)
                    train_mask[i, :start] = 0
            else:
                tail = min(7, train_set.size(1))
                train_mask = torch.zeros((max_samples, train_set.size(1)), dtype=torch.int64)
                train_mask[:, -tail:] = 1

            # --------- model / opt / sched ----------
            model = TFModel(config).to(config.device)

            # save initial weights into the p1 folder (name includes p2)
            init_name = f"initial_state_p2{fmt(p2_val)}.pt"
            torch.save(model.state_dict(), os.path.join(p1_dir, init_name))

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=config.wd if config.use_wd else 0,
            )
            scheduler = make_scheduler(optimizer, config)

            
            
            train_ds = TensorDataset(train_set, train_mask)
            train_ds = ConcatDataset([train_ds] * num_epochs)
            
            
            train_loader = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=10,
                generator=generator,
                worker_init_fn=seed_worker
            )

            config.num_epoch = min(train_set.size(0)*num_epochs/batch_size, batch_limit*num_epochs)

            # --------- wandb run per (p1, p2) ----------
            os.environ["WANDB_MODE"] = "online" if config.wandb_online else "offline"
            run = wandb.init(
                entity=getattr(config, "wandb_entity_name", None),
                project=getattr(config, "wandb_project_name", None),
                name=f"mod{modulus}_p1{fmt(p1_val)}_p2{fmt(p2_val)}",
                config={
                    "num_expressions": 3,
                    "num_layers": config.num_layers + 1,
                    "num_head": config.num_heads,
                    "batch_size": config.batch_size,
                    "learning_rate": config.lr,
                    "weight_decay": config.wd,
                    "dropout": config.dropout,
                    "epochs": num_epochs,
                    "modulus": modulus,
                    "p1": p1_val,
                    "p2": p2_val,
                },
            )

            # --------- train ----------
            model, training_info = train_with_dataloader(
                model,
                config,
                optimizer,
                scheduler,
                run,
                train_loader,
                N=1000,
                p1=p1_val,
                p2=p2_val,
                training_info=None,
                modulus=modulus,
                vocab=vocab,
            )

            # --------- save final checkpoint into the p1 folder (filename carries p2) ----------
            final_name = f"{config.num_layers+1}layers{config.num_heads}heads_p2{fmt(p2_val)}.pt"
            torch.save(model.state_dict(), os.path.join(p1_dir, final_name))
            
            
            

            # --------- save training info (keep global dir; or move under p1_dir if you prefer) ----------
            info_name = f"training_information_mod{modulus}_p1{fmt(p1_val)}_p2{fmt(p2_val)}.pt"
            torch.save(training_info, os.path.join(info_root, info_name))

            wandb.finish()
            
           
            
            # --------- release memory between runs ----------
            try:
                del model, optimizer, scheduler, train_loader, train_ds, train_set, test_set, train_mask
                torch.cuda.empty_cache()
            except Exception:
                pass

            
            
            
            
            


if __name__ == "__main__":
    fire.Fire(main)
