"""Training a surrogate to predict  state(t) -> state(t + dt), one-step or on its own rollouts.

  # sanity check: overfit a single training trajectory (a few minutes on a laptop)
  python -m surrogate.train --name overfit_resnet --overfit 0 --width 32 --depth 4 --steps 500

  # real training (on a GPU)
  python -m surrogate.train --name resnet_w64_d8 --width 64 --depth 8 --steps 20000

  # U-Net (width = channels at the finest level, doubling per level; --levels halvings)
  python -m surrogate.train --name unet_w16_l3 --model unet --width 16 --levels 3 --steps 2000
  
  # positivity-preserving model: works in (log rho, u, v, log p), can never output rho <= 0 or p <= 0
  python -m surrogate.train --name pos_w32_d4 --positive --width 32 --depth 4 --steps 2000

  # fine-tune an existing run on its own rollouts (see --unroll / --mode below)
  python -m surrogate.train --name pf3 --init runs/resnet_w32_d4_mac --unroll 3 --mode pushforward \
      --steps 600 --lr 3e-4 --warmup 20

Writes runs/<name>/: config.json, log.csv (one row per evaluation), best.eqx and last.eqx (weights).

Loss = mean squared error of the predicted next state, in normalised units. Alongside it we report
"vs persistence" = loss / loss of predicting "nothing changes". Below 1 means the model has learned
something; 0.1 means its error is 10x smaller than the trivial guess. The validation loss is always the
ONE-step loss, whatever the training mode, so runs stay comparable.

Training modes (--unroll k, --mode):
  k = 1                 plain one-step training: input = true frame t, target = frame t+1.
  multistep, k > 1      apply the model k times from the true frame t, penalise the error at EVERY step
                        (t+1 .. t+k), with gradients through the whole chain. Cost ~k x one-step.
  pushforward, k > 1    apply the model k-1 times WITHOUT gradients, then train one step from that slightly
                        wrong input to the true frame t+k. The model learns to correct its own typical
                        errors. Cost ~(1 + (k-1)/3) x one-step. (Brandstetter, Worrall & Welling, ICLR 2022)
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
from surrogate.models import Positive, ResNet, count_params, logprim_stats
from surrogate.models import Positive, ResNet, UNet, count_params, logprim_stats

POSITIVE_KEYS = ("u_mean", "u_std", "w_mean", "w_std", "w_delta_scale")  


def build_model(cfg, delta_scale, key):
    if cfg.get("positive"):
        core = build_model({**cfg, "positive": False}, cfg["w_delta_scale"], key)
        return Positive(core, tuple(cfg["u_mean"]), tuple(cfg["u_std"]), tuple(cfg["w_mean"]), tuple(cfg["w_std"]))
    if cfg["model"] == "resnet":
        return ResNet(width=cfg["width"], depth=cfg["depth"], delta_scale=delta_scale, key=key)
    if cfg["model"] == "unet":
        return UNet(width=cfg["width"], levels=cfg.get("levels", 3), delta_scale=delta_scale, key=key)
    raise ValueError(f"unknown model {cfg['model']}")


def load_model(run_dir, which="best"):
    """Rebuild a model from runs/<name>/config.json and load its saved weights."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    skeleton = build_model(cfg, cfg["delta_scale"], jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(run_dir / f"{which}.eqx", skeleton), cfg


def mse(model, x, y):
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)


def apply_n(model, x, n):
    """Apply the model n times (n is a Python int, fixed at compile time). x: (B, 4, n, n)."""
    for _ in range(n):
        x = jax.vmap(model)(x)
    return x


def loss_onestep(model, x, y):
    """x: (B, 4, n, n) true frame t; y: (B, k, 4, n, n) true frames t+1..t+k. Uses y[:, 0] only."""
    return mse(model, x, y[:, 0])


def loss_multistep(model, x, y):
    """Mean error over k self-fed steps, gradients through the whole chain.

    A plain Python loop (k is small and fixed, so JAX compiles it into one straight-line program).
    jax.lax.scan does the same maths but was ~18x slower here on the CPU backend.
    """
    losses = []
    for j in range(y.shape[1]):
        x = jax.vmap(model)(x)
        losses.append(jnp.mean((x - y[:, j]) ** 2))
    return jnp.mean(jnp.stack(losses))


def loss_pushforward(model, x, y):
    """k-1 self-fed steps with NO gradient, then one trained step to the true frame t+k."""
    k = y.shape[1]
    x_pushed = jax.lax.stop_gradient(apply_n(model, x, k - 1))
    return mse(model, x_pushed, y[:, -1])


LOSSES = {"onestep": loss_onestep, "multistep": loss_multistep, "pushforward": loss_pushforward}


def make_step(optimiser, loss_fn):
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
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
    p.add_argument("--model", default="resnet", choices=["resnet", "unet"])
    p.add_argument("--width", type=int, default=64, help="resnet: channels; unet: channels at the finest level")
    p.add_argument("--depth", type=int, default=8, help="resnet: number of residual blocks")
    p.add_argument("--levels", type=int, default=3, help="unet: number of halvings (n must divide by 2**levels)")
    p.add_argument("--n", type=int, default=64, help="training resolution (64 or 128)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-pairs", type=int, default=512, help="validation pairs per evaluation (fixed subset)")
    p.add_argument("--overfit", type=int, default=None, metavar="K",
                   help="train AND evaluate on training trajectory K only (sanity check)")
    p.add_argument("--unroll", type=int, default=1, metavar="K", help="training horizon k (1 = one-step)")
    p.add_argument("--mode", default="pushforward", choices=["multistep", "pushforward"],
                   help="how to use k > 1 steps (ignored when --unroll 1)")
    p.add_argument("--positive", action="store_true",
                   help="wrap the model so it predicts (log rho, u, v, log p): density and pressure stay positive")
    p.add_argument("--init", default=None, metavar="RUN",
                   help="start from the best weights of runs/<RUN> (fine-tuning); model settings are taken from it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data/packed")
    p.add_argument("--runs", default="runs")
    args = p.parse_args()

    out = Path(args.runs) / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)


    if args.init:                                           
        init_cfg = json.loads((Path(args.init) / "config.json").read_text())
        for key in ("model", "width", "depth", "n"):
            setattr(args, key, init_cfg[key])
        args.levels = init_cfg.get("levels", 3)
        args.positive = init_cfg.get("positive", False)
    train = PairDataset(args.data, "train", args.n, horizon=args.unroll)
    if args.overfit is not None:
        train.pairs = train.pairs[train.pairs[:, 0] == args.overfit]
        val, val_pairs = train, train.pairs
    else:
        val = PairDataset(args.data, "val", args.n, horizon=1)
        val_pairs = val.pairs[np.random.default_rng(1234).choice(len(val), min(args.val_pairs, len(val)),
                                                                  replace=False)]
    delta_scale = (train.norm.delta_std / train.norm.std).tolist()

    cfg = {**vars(args), "delta_scale": delta_scale}
    if args.positive:                       # statistics of (log rho, u, v, log p), from the TRAIN split only
        if args.init:
            cfg.update({k: init_cfg[k] for k in POSITIVE_KEYS})
        else:
            cfg.update(logprim_stats(train), u_mean=train.norm.mean.tolist(), u_std=train.norm.std.tolist())
    if args.init:
        model, _ = load_model(args.init, "best")
    else:
        model = build_model(cfg, delta_scale, jax.random.PRNGKey(args.seed))
    mode = "onestep" if args.unroll == 1 else args.mode
    cfg["mode"] = mode
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, min(args.warmup, args.steps // 2), args.steps,
                                                  end_value=args.lr * 0.01)
    optimiser = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))
    step_fn = make_step(optimiser, LOSSES[mode])

    cfg.update(params=count_params(model), train_pairs=len(train), val_pairs_used=len(val_pairs),
               device=str(jax.devices()[0]))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{args.model}: {cfg['params']:,} parameters | {len(train):,} training pairs | "
          f"batch {args.batch} | {args.steps} steps | {mode}" + (f" k={args.unroll}" if args.unroll > 1 else "")
          + (" | positive" if args.positive else "")
          + (f" | from {args.init}" if args.init else "") + f" | {cfg['device']}")
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
        model, opt_state, loss = step_fn(model, opt_state, jnp.asarray(x), jnp.asarray(y))
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