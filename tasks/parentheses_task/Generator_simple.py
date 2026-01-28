import json
import os
import math
import random
import yaml
import torch
import numpy as np
import fire
import multiprocessing as mp
from arithmetic_experiments.parentheses_task_experiment.utils import Config
from modular_data_generation import generate_noisy_twosteps_dataset_modp, generate_noisy_twosteps_dataset_modp_1, generate_noisy_twosteps_varied_dataset_modp
from arithmetic_experiments.parentheses_task_experiment.Multiplication_addition_training_simple import decode_expressions


import os, math, random, json
import numpy as np
import torch
import multiprocessing as mp


def _write_chunk_to_memmap(args):
    # args: (mm_path, dtype_str, shape_tuple, start, end, chunk_size, seed_offset, modulus, p1, p2)
    (mm_path, dtype_str, shape, start, end, chunk_size, seed_offset, modulus, p1, p2) = args



    base_seed = 42 + seed_offset
    random.seed(base_seed)
    torch.manual_seed(base_seed)


    data, _, _ = generate_noisy_twosteps_dataset_modp_1(N=chunk_size, p=modulus, p1=p1, p2=p2)
    data = data.to(torch.uint8)                  
    arr = data.numpy()                          


    mm = np.memmap(mm_path, dtype=np.dtype(dtype_str), mode='r+', shape=tuple(shape))
    mm[start:end, ...] = arr
    mm.flush()
    del mm
    return (start, end)


def build_and_save_memmap(
    total: int = 1_000_000,
    chunk_size: int = 10_000,
    num_workers: int | None = None,
    p1: float = 0.05,
    p2: float = 0.05,
    modulus: int = 11,
    save_path: str = "./3steps_multiplication_addition_dataset/mod11_0.05_0.05.pt",
    tmp_dir: str = "./_mm_tmp"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    num_chunks = math.ceil(total / chunk_size)
    sizes = [chunk_size] * (num_chunks - 1) + [total - chunk_size * (num_chunks - 1)]
    offsets = [0]
    for s in sizes: offsets.append(offsets[-1] + s)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    first_size = sizes[0]
    data0, _, _ = generate_noisy_twosteps_dataset_modp_1(N=first_size, p=modulus, p1=p1, p2=p2)
    data0 = data0.to(torch.uint8)
    sample_shape = tuple(data0.shape[1:])     # e.g. () 或 (C, H, W)


    dtype_str = 'uint8'
    mm_shape = (total, *sample_shape)
    mm_path = os.path.join(tmp_dir, f"mm_{os.getpid()}_{abs(hash(save_path)) % (10**8)}.dat")
    mm = np.memmap(mm_path, dtype=np.uint8, mode='w+', shape=mm_shape)

 
    mm[offsets[0]:offsets[1], ...] = data0.numpy()
    mm.flush()
    del mm  


    tasks = [
        (mm_path, dtype_str, mm_shape, offsets[i], offsets[i+1], sizes[i], i, modulus, p1, p2)
        for i in range(1, num_chunks)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, maxtasksperchild=16) as pool:
        for start, end in pool.imap_unordered(_write_chunk_to_memmap, tasks, chunksize=1):
            pass

    
    mm = np.memmap(mm_path, dtype=np.uint8, mode='r', shape=mm_shape)
    tensor = torch.from_numpy(np.asarray(mm))  
    torch.save(tensor.clone(), save_path)      
    del mm
    try:
        os.remove(mm_path)
    except OSError:
        pass

    print(f"[OK] Saved {tuple(tensor.shape)} tensor to: {save_path}")


def generate_grid_datasets(
    p1_grid: list = [0.001],
    p2_grid: list = [0.001],
    total: int = 2_000_000,
    chunk_size: int = 10_000,
    num_workers: int | None = 6,
    modulus: int = 11,
    target_dir: str | None = None,
):
    if target_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, "3steps_multiplication_addition_dataset")

    os.makedirs(target_dir, exist_ok=True)
    for p1 in p1_grid:
        for p2 in p2_grid:
            save_path = os.path.join(target_dir, f"mod{modulus}_{p1:.3f}_{p2:.3f}.pt")
            print(f"\n=== Generating dataset for p1={p1:.3f}, p2={p2:.3f} ===")
            build_and_save_memmap(
                total=total,
                chunk_size=chunk_size,
                num_workers=num_workers,
                p1=p1,
                p2=p2,
                modulus=modulus,
                save_path=save_path,
            )
    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    fire.Fire(generate_grid_datasets)