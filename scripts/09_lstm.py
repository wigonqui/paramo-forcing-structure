"""
Stage 09 - LSTM benchmark

Purpose
    Train the deep-learning reference structure of the factorial design: a single-catchment
    hourly LSTM for every precipitation scenario of the site, evaluated exactly like GR4H.

Methods
    Inputs: precipitation of the scenario (log1p), catchment temperature, PET and hour and
    day-of-year harmonics; the optional hybrid variant (LSTM_hybrid) also receives the GR4H
    discharge and stores (stage 08). Sequence-to-sequence training on windows of
    spinup_h + window_h hours with loss only after the spin-up, so each hour is processed about
    1.5 times per epoch instead of hundreds of times in many-to-one training; random window
    offsets per epoch; early stopping on lstm.stop_period (inside calibration) evaluated every
    validate_every epochs; an ensemble of lstm.seeds networks. Prediction uses the same
    blocks and spin-up as training. Least-squares training shrinks hydrograph variability,
    so the ensemble mean receives the variance adjustment of paramo.metrics (estimated on
    training data only); the unadjusted series is kept as LSTM_raw. Scenarios run in
    parallel processes (lstm.workers), with the CPU threads shared among them.
    The model is trained on one catchment on purpose, reproducing the data situation of a
    local water utility (see Kratzert et al., 2024, on single-basin training).

Tools
    PyTorch, pandas, numpy, concurrent.futures; paramo.metrics.

Inputs
    data/processed/<site>/forcing_<catchment>.parquet (07); simulations/GR4H__* (08)

Outputs
    data/processed/<site>/simulations/LSTM__*, LSTM_raw__* [and LSTM_hybrid__*]
    outputs/<site>/tables/09_metrics.csv, 09_training_log.csv

Usage
    python scripts/09_lstm.py
    python scripts/09_lstm.py --scenarios gauge fusion_corr --seeds 1

Next stage
    10 - GBM and GR4H-GBM.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo.core import (get_logger, in_period, load_config, path, read_simulation,
                         simulation_path, split_scenarios, write_simulation)
from paramo.metrics import evaluate, variance_adjustment

log = get_logger("09_lstm")


# --------------------------------------------------------------------------- data
def model_inputs(cfg, catchment: str, scenario: str, hybrid: bool) -> pd.DataFrame:
    forcing = pd.read_parquet(path(cfg, "processed") / f"forcing_{catchment}.parquet")
    gr = read_simulation(simulation_path(cfg, "GR4H", scenario, catchment))
    forcing = forcing.reindex(gr.index)
    x = pd.DataFrame(index=gr.index)
    x["P"] = np.log1p(forcing[f"P__{scenario}"].clip(lower=0).fillna(0))
    x["T"] = forcing["T_catchment"].interpolate(limit_direction="both")
    x["PET"] = forcing["PET"].fillna(0)
    h, d = x.index.hour.to_numpy(), x.index.dayofyear.to_numpy()
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)
    if hybrid:
        x["Q_gr4h"] = np.log1p(gr["Qsim"].clip(lower=0))
        x["Prod"], x["Rout"] = gr["Prod"], gr["Rout"]
    x["Qobs"] = gr["Qobs"]
    x["excluded"] = gr["excluded"] | gr["Qobs"].isna()
    return x


# --------------------------------------------------------------------------- network
def build_network(n_in: int, hidden: int, dropout: float):
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.head(self.drop(h)).squeeze(-1)

    return Network()


def predict(net, xn: np.ndarray, lc: dict, device, batch: int = 64) -> np.ndarray:
    """Blocks of window_h hours, each preceded by spinup_h hours; the series start is padded
    by repeating the first row so that every block has a full spin-up."""
    import torch
    w0, w = lc["spinup_h"], lc["window_h"]
    n = len(xn)
    xp = np.vstack([np.repeat(xn[:1], w0, 0), xn, np.repeat(xn[-1:], w, 0)])
    starts = np.arange(0, n, w)
    windows = starts[:, None] + np.arange(w0 + w)[None, :]
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(starts), batch):
            xb = torch.tensor(xp[windows[k:k + batch]], dtype=torch.float32, device=device)
            out.append(net(xb)[:, w0:].cpu().numpy().reshape(-1))
    return np.concatenate(out)[:n]


def train(xn, yn, train_mask, stop_mask, lc, seed, device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_network(xn.shape[1], lc["hidden"], lc["dropout"]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lc["learning_rate"])
    w0, w = lc["spinup_h"], lc["window_h"]
    xt = torch.tensor(xn, dtype=torch.float32, device=device)
    yt = torch.tensor(np.nan_to_num(yn), dtype=torch.float32, device=device)
    mt = torch.tensor(train_mask, dtype=torch.float32, device=device)
    n = len(xn)
    stop_end = int(np.flatnonzero(stop_mask)[-1]) + 1 if stop_mask.any() else n
    best, since, state, history = np.inf, 0, None, []
    for epoch in range(lc["max_epochs"]):
        net.train()
        starts = np.arange(rng.integers(0, w), n - w0 - w, w)
        starts = starts[np.array([train_mask[s + w0:s + w0 + w].any() for s in starts], bool)]
        if len(starts) == 0:
            raise ValueError("No training windows with data: check the periods")
        rng.shuffle(starts)
        losses = []
        for k in range(0, len(starts), lc["batch"]):
            ix = torch.tensor(starts[k:k + lc["batch"]], device=device)
            rows = ix[:, None] + torch.arange(w0 + w, device=device)[None, :]
            pred = net(xt[rows])[:, w0:]
            target, weight = yt[rows][:, w0:], mt[rows][:, w0:]
            loss = ((pred - target) ** 2 * weight).sum() / weight.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % lc["validate_every"] and epoch + 1 < lc["max_epochs"]:
            continue
        p = predict(net, xn[:stop_end], lc, device)
        sm = stop_mask[:stop_end]
        stop_loss = float(np.nanmean((p[sm] - yn[:stop_end][sm]) ** 2))
        history.append((epoch, float(np.mean(losses)), stop_loss))
        if stop_loss < best - 1e-4:
            best, since = stop_loss, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += lc["validate_every"]
            if since >= lc["patience"]:
                break
    if state is not None:
        net.load_state_dict(state)
    return net, history


# --------------------------------------------------------------------------- one scenario
def run_scenario(site: str, experiment: str | None, structure: str, scenario: str,
                 n_seeds: int, threads: int) -> tuple[list, list]:
    import torch
    torch.set_num_threads(threads)
    cfg = load_config(site, experiment, prepare=False)
    lc, per, gr = cfg["lstm"], cfg["periods"], cfg["gr4h"]
    cc, ct = gr["calibration_catchment"], gr["transfer_catchment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid = structure == "LSTM_hybrid"
    xc, xt = model_inputs(cfg, cc, scenario, hybrid), model_inputs(cfg, ct, scenario, hybrid)
    cols = [c for c in xc if c not in ("Qobs", "excluded")]
    valid = ~xc["excluded"].to_numpy()
    calibration = in_period(xc.index, per["calibration"])
    stop = in_period(xc.index, lc["stop_period"])
    train_mask, stop_mask = calibration & ~stop & valid, stop & valid
    mu, sd = xc.loc[train_mask, cols].mean(), xc.loc[train_mask, cols].std().replace(0, 1)
    qm, qs = xc.loc[train_mask, "Qobs"].mean(), xc.loc[train_mask, "Qobs"].std()
    xn_c = ((xc[cols] - mu) / sd).fillna(0).to_numpy()
    xn_t = ((xt[cols] - mu) / sd).fillna(0).to_numpy()
    yn = ((xc["Qobs"] - qm) / qs).to_numpy()

    pred_c, pred_t, log_rows = [], [], []
    for seed in range(n_seeds):
        net, history = train(xn_c, yn, train_mask, stop_mask, lc, 1000 + seed, device)
        pred_c.append(np.clip(predict(net, xn_c, lc, device) * qs + qm, 0, None))
        pred_t.append(np.clip(predict(net, xn_t, lc, device) * qs + qm, 0, None))
        best = min(history, key=lambda h: h[2])
        log_rows.append({"structure": structure, "scenario": scenario, "seed": seed,
                         "epochs": history[-1][0] + 1, "best_epoch": best[0] + 1,
                         "stop_mse": round(best[2], 4)})

    mean_c, mean_t = np.vstack(pred_c).mean(0), np.vstack(pred_t).mean(0)
    adjust, c = variance_adjustment(mean_c, xc["Qobs"].to_numpy(), train_mask)
    outputs = {structure: (adjust(mean_c), adjust(mean_t))}
    if structure == "LSTM":
        outputs["LSTM_raw"] = (mean_c, mean_t)
    for row in log_rows:
        row["variance_factor"] = round(c, 3)

    metric_rows = []
    for name, (sc, st) in outputs.items():
        for catchment, x, sim in ((cc, xc, sc), (ct, xt, st)):
            out = pd.DataFrame({"Qobs": x["Qobs"].to_numpy(), "Qsim": sim,
                                "excluded": x["excluded"].to_numpy()}, index=x.index)
            write_simulation(out, simulation_path(cfg, name, scenario, catchment))
            periods = ([("calibration", per["calibration"]), ("validation", per["validation"])]
                       if catchment == cc else [("transfer", per["transfer"])])
            for pname, rng_ in periods:
                for row in evaluate(out, rng_):
                    metric_rows.append({"model": name, "scenario": scenario,
                                        "catchment": catchment, "period": pname, **row})
    return metric_rows, log_rows


# --------------------------------------------------------------------------- main
def main(a):
    cfg = load_config()
    lc = cfg["lstm"]
    forcing = pd.read_parquet(path(cfg, "processed") /
                              f"forcing_{cfg['gr4h']['calibration_catchment']}.parquet")
    scenarios = a.scenarios or split_scenarios(forcing.columns)[0]
    structures = a.structures or lc["structures"]
    tasks = [(s, sc) for s in structures for sc in scenarios]
    workers = max(1, min(lc["workers"], len(tasks)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    seeds = a.seeds or lc["seeds"]
    log.info("Site %s | structures %s | %d scenarios | seeds %d | workers %d x %d threads",
             cfg["site"], structures, len(scenarios), seeds, workers, threads)
    metrics, train_log = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_scenario, cfg["site"], cfg["experiment"], s, sc, seeds,
                             threads): (s, sc) for s, sc in tasks}
        for fu in as_completed(futures):
            m, tl = fu.result()
            metrics += m
            train_log += tl
            k = {r["period"]: r["kge"] for r in m if r["scale"] == "hourly"
                 and r["model"] == futures[fu][0]}
            log.info("%-11s %-15s KGE' hourly: calibration=%.3f | validation=%.3f | "
                     "transfer=%.3f (%.1f min elapsed)", *futures[fu], k["calibration"],
                     k["validation"], k["transfer"], (time.time() - t0) / 60)
    pd.DataFrame(metrics).round(4).to_csv(path(cfg, "tables") / "09_metrics.csv", index=False)
    pd.DataFrame(train_log).to_csv(path(cfg, "tables") / "09_training_log.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--structures", nargs="*", choices=["LSTM", "LSTM_hybrid"])
    ap.add_argument("--seeds", type=int)
    main(ap.parse_args())
