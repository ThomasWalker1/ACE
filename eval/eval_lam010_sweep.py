"""
eval_lam010_sweep.py — T-sweep at λ=0.10 across all four attack conditions.

For each of FGSM/PGD × ε=4/255/8/255, sweeps T ∈ {0,5,10,15,20,25,30}
in a single pass (Adam accumulates steps; snapshots taken at each T).
Also records gap recovery and clean accuracy drop for the Pareto curve.

Output:
  JSON  → results/lam010_sweep/results.json
  CSV   → paper/figures/data/lam010_{attack}_{eps}.csv  (for pgfplots)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eval_adv_purification import ImageNetSubset, pgd, fgsm, accuracy
from ace import bpp_proxy

LAM = 0.10
T_VALS = [0, 5, 10, 15, 20, 25, 30]


def purify_sweep(x_adv, x_clean, T_checkpoints, lr=0.01):
    """Run Adam to max(T_checkpoints); snapshot x_adv and x_clean at each T."""
    max_T = max(T_checkpoints)
    ckpts = sorted(set(T_checkpoints))

    xa = x_adv.clone().detach().requires_grad_(True)
    xc = x_clean.clone().detach().requires_grad_(True)
    oa = torch.optim.Adam([xa], lr=lr)
    oc = torch.optim.Adam([xc], lr=lr)

    snaps = {}
    if 0 in ckpts:
        snaps[0] = (x_adv.clone(), x_clean.clone())

    for t in range(1, max_T + 1):
        oa.zero_grad()
        xac = xa.clamp(0.0, 1.0)
        (F.mse_loss(xac, x_adv) + LAM * bpp_proxy(xac).mean()).backward()
        oa.step()

        oc.zero_grad()
        xcc = xc.clamp(0.0, 1.0)
        (F.mse_loss(xcc, x_clean) + LAM * bpp_proxy(xcc).mean()).backward()
        oc.step()

        if t in ckpts:
            snaps[t] = (xa.detach().clamp(0.0, 1.0).clone(), xc.detach().clamp(0.0, 1.0).clone())
    return snaps


def run_attack(model, loader, attack_fn, device):
    """Returns dict {T: {defended_acc, clean_pur_acc, n}} + baseline accs."""
    stats = {T: [0.0, 0.0, 0] for T in T_VALS}
    clean_c = adv_c = 0.0
    total_n = 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = attack_fn(model, x, y)

        with torch.no_grad():
            clean_c += accuracy(model, x, y) * x.shape[0]
            adv_c += accuracy(model, x_adv, y) * x.shape[0]

        with torch.enable_grad():
            snaps = purify_sweep(x_adv, x, T_VALS)

        for T, (xa_pur, xc_pur) in snaps.items():
            with torch.no_grad():
                stats[T][0] += accuracy(model, xa_pur, y) * x.shape[0]
                stats[T][1] += accuracy(model, xc_pur, y) * x.shape[0]
                stats[T][2] += x.shape[0]
        total_n += x.shape[0]

    clean_acc = clean_c / total_n * 100
    adv_acc = adv_c / total_n * 100
    gap = max(clean_acc - adv_acc, 1e-6)

    rows = []
    for T in T_VALS:
        def_a, cln_a, n = stats[T]
        def_acc = def_a / n * 100
        cln_acc = cln_a / n * 100
        recovery_pct = round((def_acc - adv_acc) / gap * 100, 1)
        assert -1.0 <= recovery_pct <= 101.0, f"recovery_pct out of bounds at T={T}: {recovery_pct}"
        rows.append(
            {
                "T": T,
                "defended_acc": round(def_acc, 2),
                "clean_pur_acc": round(cln_acc, 2),
                "recovery_pct": recovery_pct,
                "clean_drop_pp": round(clean_acc - cln_acc, 2),
            }
        )

    best = max(rows, key=lambda r: r["recovery_pct"])
    return {
        "clean_acc": round(clean_acc, 2),
        "adv_acc": round(adv_acc, 2),
        "rows": rows,
        "best_T": best["T"],
        "best_recovery": best["recovery_pct"],
        "best_clean_drop": best["clean_drop_pp"],
        "best_defended_acc": best["defended_acc"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_images", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", default="results/lam010_sweep")
    p.add_argument("--csv_dir", default="results/lam010_sweep/figure_data")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    model = models.resnet50(weights="IMAGENET1K_V1").to(device).eval()
    dataset = ImageNetSubset(args.data_dir, args.n_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    attacks = [
        ("fgsm", 4, lambda m, x, y: fgsm(m, x, y, 4 / 255)),
        ("fgsm", 8, lambda m, x, y: fgsm(m, x, y, 8 / 255)),
        ("pgd", 4, lambda m, x, y: pgd(m, x, y, 4 / 255, steps=10, alpha=4 / 255 / 4)),
        ("pgd", 8, lambda m, x, y: pgd(m, x, y, 8 / 255, steps=10, alpha=8 / 255 / 4)),
    ]

    all_results = {}
    for atk_name, eps_int, atk_fn in attacks:
        key = f"{atk_name}_eps{eps_int}"
        print(f"\n── {key.upper()}  λ={LAM} ──")
        res = run_attack(model, loader, atk_fn, device)
        all_results[key] = res

        print(f"  Clean {res['clean_acc']:.2f}%  Adv {res['adv_acc']:.2f}%")
        print(
            f"  Best T={res['best_T']}: defended {res['best_defended_acc']:.2f}%  "
            f"recovery {res['best_recovery']:.1f}%  "
            f"clean_drop {res['best_clean_drop']:.1f}pp"
        )

        # Save CSV for pgfplots (gap recovery vs clean drop)
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"lam010_{key}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["T", "recovery_pct", "clean_drop_pp"])
            w.writeheader()
            for r in res["rows"]:
                w.writerow({k: r[k] for k in ["T", "recovery_pct", "clean_drop_pp"]})
        print(f"  CSV → {csv_path}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")


if __name__ == "__main__":
    main()
