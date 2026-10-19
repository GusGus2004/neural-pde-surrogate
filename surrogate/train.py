"""One-step training of surrogate: learn  state(t) -> state(t + dt).

Run from repo root

  # check: overfit a single training trajectory
  python -m surrogate.train --name overfit_resnet --overfit 0 --width 32 --depth 4 --steps 500

  # real training (on a GPU)
  python -m surrogate.train --name resnet_w64_d8 --width 64 --depth 8 --steps 20000

Writes runs/<name>/: config.json, log.csv (one row per evaluation), best.eqx and last.eqx (weights).

Loss = mean squared error of the predicted next state, in normalised units. Alongside it report
"vs persistence" = loss / loss of predicting "nothing changes". Below 1 means the model has learned
something; 0.1 means its error is 10x smaller than the trivial guess.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from surrogate.data import PairDataset
from surrogate.models import ResNet, count_params


def build_model(cfg, delta_scale, key):
    if cfg["model"] == "resnet":
        return ResNet(width=cfg["width"], depth=cfg["depth"], delta_scale=delta_scale, key=key)
    raise ValueError(f"unknown model {cfg['model']}")


def load_model(run_dir, which="best"):
    """Rebuild a model from runs/<name>/config.json and load its saved weights."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    skeleton = build_model(cfg, cfg["delta_scale"], jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(run_dir / f"{which}.eqx", skeleton), cfg


def mse(model, x, y):
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)


def make_step(optimiser):
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(mse)(model, x, y)
        updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss
    return step


@eqx.filter_jit
def eval_batch(model, x, y):
    """Sum of squared errors of the model and of persistence (predict x itself), for averaging later."""
    pred = jax.vmap(model)(x)
    return jnp.sum((pred - y) ** 2), jnp.sum((x - y) ** 2)


def evaluate(model, ds, pairs, batch_size):
    """Mean squared error over the given pairs, and the same for persistence."""
    se_model = se_persist = 0.0
    for i in range(0, len(pairs), batch_size):
        x, y = ds.get(pairs[i:i + batch_size])
        a, b = eval_batch(model, jnp.asarray(x), jnp.asarray(y[:, 0]))
        se_model += float(a)
        se_persist += float(b)
    count = len(pairs) * 4 * ds.n * ds.n
    return se_model / count, se_persist / count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name: output goes to runs/<name>/")
    p.add_argument("--model", default="resnet")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--n", type=int, default=64, help="training resolution (64 or 128)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-pairs", type=int, default=512, help="validation pairs per evaluation (fixed subset)")
    p.add_argument("--overfit", type=int, default=None, metavar="K",
                   help="train AND evaluate on training trajectory K only (sanity check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data/packed")
    p.add_argument("--runs", default="runs")
    args = p.parse_args()

    out = Path(args.runs) / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Data
    train = PairDataset(args.data, "train", args.n, horizon=1)
    if args.overfit is not None:
        train.pairs = train.pairs[train.pairs[:, 0] == args.overfit]
        val, val_pairs = train, train.pairs
    else:
        val = PairDataset(args.data, "val", args.n, horizon=1)
        val_pairs = val.pairs[np.random.default_rng(1234).choice(len(val), min(args.val_pairs, len(val)),
                                                                  replace=False)]
    delta_scale = (train.norm.delta_std / train.norm.std).tolist()

    # Model and optimiser: AdamW, linear warmup then cosine decay, gradient clipping
    cfg = {**vars(args), "delta_scale": delta_scale}
    model = build_model(cfg, delta_scale, jax.random.PRNGKey(args.seed))
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, min(args.warmup, args.steps // 2), args.steps,
                                                  end_value=args.lr * 0.01)
    optimiser = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))
    step_fn = make_step(optimiser)

    cfg.update(params=count_params(model), train_pairs=len(train), val_pairs_used=len(val_pairs),
               device=str(jax.devices()[0]))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{args.model}: {cfg['params']:,} parameters | {len(train):,} training pairs | "
          f"batch {args.batch} | {args.steps} steps | {cfg['device']}")
    print(f"{'step':>6} {'train':>10} {'val':>10} {'vs persist':>10} {'lr':>9} {'s/step':>7}")

    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "val_loss", "val_persistence", "val_ratio", "lr", "elapsed_s"])

    def batches_forever():
        while True:
            yield from train.batches(min(args.batch, len(train)), rng=rng)

    best = np.inf
    t_start = t_last = time.time()
    running = []
    for step, (x, y) in enumerate(batches_forever(), start=1):
        model, opt_state, loss = step_fn(model, opt_state, jnp.asarray(x), jnp.asarray(y[:, 0]))
        running.append(loss)
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = float(jnp.mean(jnp.stack(running)))      # syncs with the device once per eval
            running = []
            val_loss, persist = evaluate(model, val, val_pairs, args.batch)
            per_step = (time.time() - t_last) / args.eval_every
            t_last = time.time()
            lr = float(schedule(step))
            print(f"{step:>6} {train_loss:>10.3e} {val_loss:>10.3e} {val_loss / persist:>10.4f} "
                  f"{lr:>9.2e} {per_step:>7.3f}", flush=True)
            writer.writerow([step, train_loss, val_loss, persist, val_loss / persist, lr, time.time() - t_start])
            log.flush()
            if val_loss < best:
                best = val_loss
                eqx.tree_serialise_leaves(out / "best.eqx", model)
        if step == args.steps:
            break

    eqx.tree_serialise_leaves(out / "last.eqx", model)
    log.close()
    print(f"\ndone in {time.time() - t_start:.0f} s; best val loss {best:.3e}; saved to {out}/")


if __name__ == "__main__":
    main()