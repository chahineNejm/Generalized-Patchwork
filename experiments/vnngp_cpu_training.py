"""Explicit CPU VNNGP continuation, preserving the minibatch and optimizer state."""

import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import gpytorch
import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.training_control import TrainingLossStopper
from pwgp.models.vnngp import strict_gpytorch


def random_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), *state[2:]]}


def restore_random(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), pos, has_gauss, cached))


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_chunk(model, y, config, *, stopper, checkpoint, identity, resume, deadline,
                checkpoint_seconds, description, progress_every_epochs, on_epoch,
                clock=time.monotonic, max_batches=None):
    """Resume only when explicitly requested; pause after an optimizer step."""
    if config.model != "vnngp" or config.device != "cpu":
        raise ValueError("CPU continuation requires model='vnngp', device='cpu'")
    if not 0 <= config.warmup_epochs < config.epochs:
        raise ValueError("Require 0 <= warm-up < maximum epochs")
    if not np.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ValueError("checkpoint_seconds must be finite and positive")
    if type(progress_every_epochs) is not int or progress_every_epochs < 1:
        raise ValueError("progress_every_epochs must be a positive integer")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be a positive explicit test limit")
    strategy = model.variational_strategy
    optimizer = torch.optim.Adam([
        {"params": list(model.variational_parameters()), "lr": config.learning_rate_q},
        {"params": list(model.hyperparameters()), "lr": 0.0}])
    objective = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=len(y))
    losses, batch, accumulated, seen, elapsed = [], 0, 0.0, 0, 0.0
    control = {"config": asdict(stopper.config), "warmup_epochs": stopper.warmup_epochs}
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["format"] != 1:
            raise ValueError("Unsupported CPU checkpoint format")
        if state["identity"] != identity or state["config"] != asdict(config) or state["control"] != control:
            raise ValueError("Checkpoint source/data/config/stopping identity mismatch")
        if not torch.equal(state["model"]["variational_strategy.inducing_points"], strategy.inducing_points):
            raise ValueError("Checkpoint inducing locations/order differ from rebuilt model")
        if not torch.equal(state["model"]["variational_strategy.nn_xinduce_idx"], strategy.nn_xinduce_idx):
            raise ValueError("Checkpoint neighbor graph differs from rebuilt exact graph")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        iterator = state["iterator"]
        strategy._training_indices_iterator = tuple(iterator["batches"])
        strategy._training_indices_iter = iterator["cursor"]
        strategy._total_training_batches = len(iterator["batches"])
        stopper.__dict__.update(state["stopper"])
        losses, batch, accumulated, seen, elapsed = (
            state[name] for name in ("losses", "batch", "accumulated", "seen", "training_seconds"))
        if stopper.last_epoch != len(losses) or not 0 <= batch < strategy._total_training_batches:
            raise ValueError("Checkpoint epoch/minibatch/stopping counters are inconsistent")
        restore_random(state["rng"])
    elif Path(checkpoint).exists():
        raise FileExistsError("Fresh training cannot overwrite an existing checkpoint")
    started = last_saved = clock()
    processed = 0

    def save():
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite VNNGP parameter after optimization")
        stop_state = {key: value for key, value in stopper.__dict__.items() if key != "config"}
        atomic_save({"format": 1, "identity": identity, "config": asdict(config), "control": control,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": random_state(),
            "iterator": {"batches": list(strategy._training_indices_iterator),
                         "cursor": strategy._training_indices_iter},
            "stopper": stop_state, "losses": losses, "batch": batch, "accumulated": accumulated,
            "seen": seen, "training_seconds": elapsed + clock() - started}, checkpoint)

    with tqdm(total=config.epochs, initial=len(losses), desc=description, unit="epoch",
              miniters=progress_every_epochs, mininterval=0, maxinterval=float("inf")) as bar:
        while len(losses) < config.epochs and not stopper.stopped:
            if clock() >= deadline:
                save()
                return False
            model.train()
            if len(losses) >= config.warmup_epochs:
                optimizer.param_groups[1]["lr"] = config.learning_rate_hyperparameters
            optimizer.zero_grad()
            with strict_gpytorch():
                prediction = model(None)
                indices = strategy.current_training_indices
                loss = -objective(prediction, y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite VNNGP objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite VNNGP gradient")
            optimizer.step()
            accumulated += float(loss.detach()) * len(indices)
            seen += len(indices)
            batch += 1
            processed += 1
            if batch == strategy._total_training_batches:
                if seen != len(y):
                    raise ValueError("VNNGP epoch must visit every training observation exactly once")
                losses.append(accumulated / seen)
                batch, accumulated, seen = 0, 0.0, 0
                stopper.update(len(losses), losses[-1])
                on_epoch(len(losses), losses, model)
                bar.set_postfix(loss=losses[-1], refresh=False)
                bar.update(1)
            if clock() - last_saved >= checkpoint_seconds or (max_batches is not None and processed >= max_batches):
                save()
                last_saved = clock()
            if max_batches is not None and processed >= max_batches:
                return len(losses) == config.epochs or stopper.stopped
        save()
    return True


def make_stopper(values, warmup_epochs):
    from experiments.training_control import EarlyStoppingConfig

    return TrainingLossStopper(EarlyStoppingConfig(**values), warmup_epochs=warmup_epochs)
