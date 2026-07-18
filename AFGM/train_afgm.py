# -*- coding: utf-8 -*-


import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
import platform
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

try:
    from project_dir import project_dir
except Exception:
    project_dir = os.getcwd()

from util.tools import load_resnet18, get_latent_output
from util.data_loader import get_Dataloader
from util.attention import attention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRANCH_TO_INDEX = {"S": 0, "T": 1, "C": 2}


def safe_difference(a: float, b: float) -> float:
    """Return a-b only when both values are finite; otherwise return NaN."""
    if math.isfinite(a) and math.isfinite(b):
        return a - b
    return float("nan")


def safe_min_tc_minus_s(gates: List[float]) -> float:
    """Return min(T, C)-S only for the intact three-branch model."""
    gate_s, gate_t, gate_c = gates
    if all(math.isfinite(value) for value in (gate_s, gate_t, gate_c)):
        return min(gate_t, gate_c) - gate_s
    return float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.exists(path):
        return path

    candidate = os.path.join(project_dir, path)
    return candidate if os.path.exists(candidate) else path


def parse_ablation(value: str) -> List[str]:
    value = (value or "none").upper()
    if value == "NONE":
        return ["S", "T", "C"]
    if value not in {"S", "T", "C"}:
        raise ValueError("--ablate must be one of none/S/T/C.")
    return [name for name in ["S", "T", "C"] if name != value]


def count_classes(train_root: str) -> int:
    return len(
        [
            name
            for name in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, name))
        ]
    )


def make_loader(
    shape_root: str,
    texture_root: str,
    color_root: str,
    batch_size: int,
    shuffle: bool,
):

    try:
        return get_Dataloader(
            shape_root,
            texture_root,
            color_root,
            batch_size,
            shuffle=shuffle,
        )
    except TypeError:
        return get_Dataloader(
            shape_root,
            texture_root,
            color_root,
            batch_size,
        )


def to_feature_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


@torch.no_grad()
def extract_latent(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    latent = get_latent_output(model, image, "resnet18")
    return to_feature_tensor(latent, DEVICE)


def build_features(
    used: List[str],
    backbones: Dict[str, nn.Module],
    texture: torch.Tensor,
    shape: torch.Tensor,
    color: torch.Tensor,
):
    latent_shape = (
        extract_latent(backbones["S"], shape)
        if "S" in used
        else None
    )
    latent_texture = (
        extract_latent(backbones["T"], texture)
        if "T" in used
        else None
    )
    latent_color = (
        extract_latent(backbones["C"], color)
        if "C" in used
        else None
    )
    return latent_shape, latent_texture, latent_color


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if (
                key not in self.shadow
                or self.shadow[key].shape != value.shape
                or self.shadow[key].dtype != value.dtype
            ):
                self.shadow[key] = value.detach().clone()
            elif value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(),
                    alpha=1.0 - self.decay,
                )
            else:
                self.shadow[key] = value.detach().clone()

    @contextlib.contextmanager
    def apply(self, model: nn.Module):
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def configure_biological_prior(
    model: attention,
    enabled: bool,
    boost_factor: float,
) -> None:

    with torch.no_grad():
        if enabled:
            prior = torch.tensor(
                [
                    -0.05 * float(boost_factor),
                    +0.08 * float(boost_factor),
                    0.0,
                ],
                device=model.gate_prior.device,
                dtype=model.gate_prior.dtype,
            )
        else:
            prior = torch.zeros(
                3,
                device=model.gate_prior.device,
                dtype=model.gate_prior.dtype,
            )

        model.gate_prior.copy_(prior)

    print(
        "[INFO] fixed gate prior [S,T,C] = "
        f"{model.gate_prior.detach().cpu().tolist()}"
    )


def regularization_schedule(
    epoch: int,
    total_epochs: int,
    target_temperature: float,
    target_aux_weight: float,
    phased_training: bool,
) -> Tuple[float, float]:

    if not phased_training:
        return target_temperature, target_aux_weight

    if epoch < 5:
        return 1.05, 0.0

    if epoch < 15:
        progress = (epoch - 5 + 1) / 10.0
        temperature = 1.05 + progress * (
            target_temperature - 1.05
        )
        aux_weight = progress * target_aux_weight
        return temperature, aux_weight

    return target_temperature, target_aux_weight


def staged_lr_factor(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    lr_drop_epoch: int,
    stage2_lr_scale: float,
    min_lr_ratio: float,
) -> float:

    if epoch < warmup_epochs:
        return float(epoch + 1) / max(1, warmup_epochs)

    if epoch < lr_drop_epoch:
        return 1.0

    remaining = max(1, total_epochs - lr_drop_epoch)
    progress = min(
        1.0,
        max(0.0, (epoch - lr_drop_epoch) / remaining),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return (
        float(min_lr_ratio)
        + (float(stage2_lr_scale) - float(min_lr_ratio)) * cosine
    )


@torch.no_grad()
def evaluate_once(
    class_num: int,
    backbones: Dict[str, nn.Module],
    model: attention,
    loader,
    used: List[str],
    criterion: nn.Module,
    tta: bool,
) -> Tuple[float, float, float, List[float]]:
    model.eval()

    for backbone in backbones.values():
        backbone.eval()

    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    total_loss = 0.0
    total_batches = 0

    class_correct = torch.zeros(class_num, dtype=torch.long)
    class_total = torch.zeros(class_num, dtype=torch.long)


    gate_sum = torch.zeros(3, dtype=torch.float64)
    gate_count = torch.zeros(3, dtype=torch.long)

    for texture, shape, color, labels, _ in loader:
        texture = texture.to(DEVICE, non_blocking=True)
        shape = shape.to(DEVICE, non_blocking=True)
        color = color.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        features = build_features(
            used,
            backbones,
            texture,
            shape,
            color,
        )
        logits = model(*features)

        batch_gates = model.get_gate_batch()
        if batch_gates is not None:
            if batch_gates.ndim != 2:
                raise RuntimeError(
                    f"Expected gate tensor [batch, active_branches], got "
                    f"shape={tuple(batch_gates.shape)}"
                )
            if batch_gates.size(1) != len(used):
                raise RuntimeError(
                    "Gate/ablation mismatch: "
                    f"gate columns={batch_gates.size(1)}, used={used}."
                )


            for local_index, branch_name in enumerate(used):
                branch_values = batch_gates[:, local_index]
                finite_mask = torch.isfinite(branch_values)
                if finite_mask.any():
                    global_index = BRANCH_TO_INDEX[branch_name]
                    gate_sum[global_index] += (
                        branch_values[finite_mask].double().sum().cpu()
                    )
                    gate_count[global_index] += int(finite_mask.sum().item())

        if tta:
            flipped_features = build_features(
                used,
                backbones,
                torch.flip(texture, dims=[-1]),
                torch.flip(shape, dims=[-1]),
                torch.flip(color, dims=[-1]),
            )
            flipped_logits = model(*flipped_features)
            logits = (logits + flipped_logits) / 2.0

        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        total_batches += 1

        k = min(5, class_num)
        prediction = logits.topk(k, dim=1).indices

        total_samples += labels.size(0)
        top1_correct += int(
            (prediction[:, 0] == labels).sum().item()
        )
        top5_correct += int(
            (
                prediction == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
        )

        for class_index in range(class_num):
            mask = labels == class_index
            count = int(mask.sum().item())
            if count > 0:
                class_total[class_index] += count
                class_correct[class_index] += int(
                    (
                        prediction[mask, 0] == labels[mask]
                    ).sum().item()
                )

    per_class_accuracy = (
        class_correct.float()
        / class_total.clamp_min(1).float()
    ).tolist()

    mean_gates = [float("nan"), float("nan"), float("nan")]
    for branch_name, global_index in BRANCH_TO_INDEX.items():
        if int(gate_count[global_index].item()) > 0:
            mean_gates[global_index] = float(
                gate_sum[global_index].item()
                / gate_count[global_index].item()
            )

    return (
        top1_correct / max(1, total_samples),
        top5_correct / max(1, total_samples),
        total_loss / max(1, total_batches),
        per_class_accuracy,
        mean_gates,
    )


def evaluate(
    class_num: int,
    backbones: Dict[str, nn.Module],
    model: attention,
    loader,
    used: List[str],
    criterion: nn.Module,
    tta: bool,
    ema: Optional[EMA],
):
    if ema is None:
        return evaluate_once(
            class_num,
            backbones,
            model,
            loader,
            used,
            criterion,
            tta,
        )

    with ema.apply(model):
        return evaluate_once(
            class_num,
            backbones,
            model,
            loader,
            used,
            criterion,
            tta,
        )


def save_attention_checkpoint(
    path: str,
    model: attention,
    epoch: int,
    top1: float,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "val_top1": float(top1),
            "state_dict": (
                model.state_dict()
                if state_dict is None
                else state_dict
            ),
        },
        path,
    )


def main() -> None:
    script_start_time = time.perf_counter()
    args = build_parser().parse_args()
    set_seed(args.seed)

    print("[INFO] Fixed texture prior: controlled by --texture_enhance (default ON)")
    print("[INFO] Explicit texture-over-color ranking: DISABLED (lam_texture=0)")

    for field in (
        "root_shape",
        "root_texture",
        "root_color",
        "shape_model",
        "texture_model",
        "color_model",
        "save_model_dir",
        "resume",
    ):
        setattr(args, field, fix_path(getattr(args, field)))

    used = parse_ablation(args.ablate)
    used_tag = "".join(used)
    removed_branch = None if args.ablate.lower() == "none" else args.ablate.upper()
    print(
        f"[INFO] Ablation configuration: removed={removed_branch or 'NONE'}, "
        f"active={used}, output_tag={used_tag}"
    )

    train_shape = os.path.join(args.root_shape, "train")
    train_texture = os.path.join(args.root_texture, "train")
    train_color = os.path.join(args.root_color, "train")
    test_shape = os.path.join(args.root_shape, "test")
    test_texture = os.path.join(args.root_texture, "test")
    test_color = os.path.join(args.root_color, "test")

    class_num = count_classes(train_shape)

    backbones = {
        "S": load_resnet18(
            class_num,
            args.shape_model,
        ).to(DEVICE),
        "T": load_resnet18(
            class_num,
            args.texture_model,
        ).to(DEVICE),
        "C": load_resnet18(
            class_num,
            args.color_model,
        ).to(DEVICE),
    }

    # The submitted AFGM uses frozen modality encoders.
    for backbone in backbones.values():
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    train_loader = make_loader(
        train_shape,
        train_texture,
        train_color,
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        test_shape,
        test_texture,
        test_color,
        args.batch_size,
        shuffle=False,
    )

    model = attention(
        channel=len(used),
        class_num=class_num,
        multi_layer=args.multi_layer_head,
        hidden_dim=args.hidden_dim,
        first_bias=False,
        last_bias=False,
        temperature=args.temperature,
        gate_hidden=args.gate_hidden,
        gate_dropout=args.gate_dropout,
        classifier_dropout=args.classifier_dropout,
        color_feat_dropout=args.color_feat_dropout,
        branch_drop_p=args.branch_drop_p,
        branch_dropout_p=(
            args.branch_drop_s,
            args.branch_drop_t,
            args.branch_drop_c,
        ),
        feature_l2_norm=args.feature_l2_norm,
        gate_min=args.gate_min,
        gate_max=args.gate_max,
        init_gate=(1.0, 1.0, 1.0),
        init_branch_scale=(1.0, 1.0, 1.0),
    ).to(DEVICE)

    configure_biological_prior(
        model,
        enabled=args.texture_enhance,
        boost_factor=args.texture_boost_factor,
    )


    texture, shape, color, _, _ = next(iter(train_loader))
    texture = texture.to(DEVICE)
    shape = shape.to(DEVICE)
    color = color.to(DEVICE)

    with torch.no_grad():
        dummy_features = build_features(
            used,
            backbones,
            texture,
            shape,
            color,
        )
        _ = model(*dummy_features)

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume}"
            )
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
        )
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint,
        )
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )
        print(
            f"[INFO] resumed from {args.resume}; "
            f"missing={missing}, unexpected={unexpected}"
        )


    classifier_params = list(model.classifier.parameters())
    gate_params = (
        list(model.bn_dict.parameters())
        + list(model.mlp_dict.parameters())
    )

    optimizer = optim.AdamW(
        [
            {
                "params": classifier_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": gate_params,
                "lr": args.lr * args.gate_lr_scale,
                "weight_decay": args.weight_decay,
            },
        ],
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: staged_lr_factor(
            epoch=epoch,
            total_epochs=args.epoch,
            warmup_epochs=args.warmup_epochs,
            lr_drop_epoch=args.lr_drop_epoch,
            stage2_lr_scale=args.stage2_lr_scale,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(DEVICE)

    ema = EMA(
        model,
        decay=args.ema_decay,
    ) if args.ema else None

    run_name = f"AFGM_{used_tag}"
    if args.tta:
        run_name += "_TTA"

    save_dir = os.path.join(
        args.save_model_dir,
        run_name,
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            ensure_ascii=False,
            indent=2,
        )

    metrics_path = os.path.join(
        save_dir,
        "metrics_epoch.csv",
    )
    with open(
        metrics_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_cls_loss",
                "train_gate_penalty",
                "effective_aux_weight",
                "temperature",
                "raw_val_loss",
                "raw_val_top1",
                "raw_val_top5",
                "ema_val_loss",
                "ema_val_top1",
                "ema_val_top5",
                "lr",
                "gate_S",
                "gate_T",
                "gate_C",
                "texture_minus_color",
                "texture_minus_shape",
                "color_minus_shape",
                "train_seconds",
                "validation_seconds",
                "epoch_wall_seconds",
                "cumulative_wall_seconds",
                "peak_train_cuda_memory_MB",
            ]
        )

    best_raw_top1 = -1.0
    best_ema_top1 = -1.0
    best_epoch = -1
    time_to_best_wall_seconds = float("nan")
    epochs_without_improvement = 0
    completed_epochs = 0
    total_train_seconds = 0.0
    total_validation_seconds = 0.0
    max_peak_train_memory_mb = 0.0
    training_loop_start = time.perf_counter()

    for epoch in range(args.epoch):
        epoch_wall_start = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(DEVICE)
        train_start = time.perf_counter()

        model.train()

        for backbone in backbones.values():
            backbone.eval()


        gate_trainable = epoch >= args.gate_warmup_epochs
        model.set_gate_trainable(gate_trainable)

        if epoch == args.gate_warmup_epochs:
            print(
                f"[INFO] independent gates unfrozen at epoch {epoch}; "
                f"gate_lr={args.lr * args.gate_lr_scale:g}"
            )

        temperature, effective_aux = regularization_schedule(
            epoch,
            args.epoch,
            args.temperature,
            args.aux_weight,
            args.phased_training,
        )
        model.temperature = float(temperature)

        running_cls_loss = 0.0
        running_gate_penalty = 0.0
        batch_count = 0

        for iteration, (
            texture,
            shape,
            color,
            labels,
            _,
        ) in enumerate(train_loader):
            texture = texture.to(
                DEVICE,
                non_blocking=True,
            )
            shape = shape.to(
                DEVICE,
                non_blocking=True,
            )
            color = color.to(
                DEVICE,
                non_blocking=True,
            )
            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            features = build_features(
                used,
                backbones,
                texture,
                shape,
                color,
            )
            logits = model(*features)

            classification_loss = criterion(
                logits,
                labels,
            )

            if effective_aux > 0:
                gate_penalty = model.gate_penalty(
                    margin=args.shape_margin,
                    tolerance=args.tc_tolerance,
                    lam_shape=args.lam_shape,
                    lam_tc=args.lam_tc,
                    texture_margin=0.0,
                    lam_texture=0.0,
                    lam_budget=args.gate_budget_weight,
                    budget_target=args.gate_budget_target,
                )
            else:
                gate_penalty = (
                    next(model.classifier.parameters()).sum() * 0.0
                )

            total_loss = (
                classification_loss
                + effective_aux * gate_penalty
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            if ema is not None:
                ema.update(model)

            running_cls_loss += float(
                classification_loss.detach().item()
            )
            running_gate_penalty += float(
                gate_penalty.detach().item()
            )
            batch_count += 1

            if iteration % 20 == 0:
                print(
                    f"[train] epoch={epoch:02d} "
                    f"iter={iteration:04d} "
                    f"cls={running_cls_loss/max(1,batch_count):.4f} "
                    f"gate={running_gate_penalty/max(1,batch_count):.4f} "
                    f"aux={effective_aux:.4f} "
                    f"T={temperature:.3f}"
                )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        total_train_seconds += train_seconds
        peak_train_memory_mb = (
            torch.cuda.max_memory_allocated(DEVICE) / 1024.0 / 1024.0
            if DEVICE.type == "cuda"
            else float("nan")
        )
        if DEVICE.type == "cuda":
            max_peak_train_memory_mb = max(
                max_peak_train_memory_mb, peak_train_memory_mb
            )

        scheduler.step()

        validation_start = time.perf_counter()
        raw_top1, raw_top5, raw_loss, _, raw_gates = evaluate(
            class_num,
            backbones,
            model,
            val_loader,
            used,
            criterion,
            tta=args.tta,
            ema=None,
        )

        if ema is not None:
            ema_top1, ema_top5, ema_loss, _, _ = evaluate(
                class_num,
                backbones,
                model,
                val_loader,
                used,
                criterion,
                tta=args.tta,
                ema=ema,
            )
        else:
            ema_top1 = float("nan")
            ema_top5 = float("nan")
            ema_loss = float("nan")

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - validation_start
        total_validation_seconds += validation_seconds
        epoch_wall_seconds = time.perf_counter() - epoch_wall_start
        cumulative_wall_seconds = time.perf_counter() - training_loop_start
        completed_epochs = epoch + 1

        gates = raw_gates
        texture_minus_color = safe_difference(gates[1], gates[2])
        texture_minus_shape = safe_difference(gates[1], gates[0])
        color_minus_shape = safe_difference(gates[2], gates[0])
        min_tc_minus_shape = safe_min_tc_minus_s(gates)
        learning_rate = optimizer.param_groups[0]["lr"]

        with open(
            metrics_path,
            "a",
            encoding="utf-8",
            newline="",
        ) as file:
            csv.writer(file).writerow(
                [
                    epoch,
                    running_cls_loss / max(1, batch_count),
                    running_gate_penalty / max(1, batch_count),
                    effective_aux,
                    temperature,
                    raw_loss,
                    raw_top1,
                    raw_top5,
                    ema_loss,
                    ema_top1,
                    ema_top5,
                    learning_rate,
                    gates[0],
                    gates[1],
                    gates[2],
                    texture_minus_color,
                    texture_minus_shape,
                    color_minus_shape,
                    train_seconds,
                    validation_seconds,
                    epoch_wall_seconds,
                    cumulative_wall_seconds,
                    peak_train_memory_mb,
                ]
            )

        with open(
            os.path.join(save_dir, "log.txt"),
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                f"epoch={epoch} "
                f"raw_top1={raw_top1:.6f} "
                f"raw_top5={raw_top5:.6f} "
                f"ema_top1={ema_top1:.6f} "
                f"ema_top5={ema_top5:.6f} "
                f"gates={gates} "
                f"train_seconds={train_seconds:.3f} "
                f"validation_seconds={validation_seconds:.3f} "
                f"peak_train_cuda_memory_MB={peak_train_memory_mb:.3f}\n"
            )

        print(
            f"[E{epoch:02d}] "
            f"raw_top1={raw_top1:.4f} "
            f"raw_top5={raw_top5:.4f} "
            f"ema_top1={ema_top1:.4f} "
            f"gates={np.round(gates, 3)} "
            f"T-C={texture_minus_color:+.4f} "
            f"min(T,C)-S={min_tc_minus_shape:+.4f} "
            f"train={train_seconds:.1f}s val={validation_seconds:.1f}s "
            f"peak_train_mem={peak_train_memory_mb:.1f}MB"
        )

        if raw_top1 > best_raw_top1 + args.early_stop_min_delta:
            best_raw_top1 = raw_top1
            best_epoch = epoch
            time_to_best_wall_seconds = cumulative_wall_seconds
            epochs_without_improvement = 0
            save_attention_checkpoint(
                os.path.join(save_dir, "best.pth"),
                model,
                epoch,
                raw_top1,
            )
            print(
                f"  -> new best RAW Top-1={best_raw_top1:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if ema is not None and ema_top1 >= best_ema_top1:
            best_ema_top1 = ema_top1
            save_attention_checkpoint(
                os.path.join(save_dir, "best_ema.pth"),
                model,
                epoch,
                ema_top1,
                state_dict=ema.shadow,
            )
            print(
                f"  -> new best EMA Top-1={best_ema_top1:.4f}"
            )

        if args.save_ckpt:
            save_attention_checkpoint(
                os.path.join(
                    save_dir,
                    f"model_ck_{epoch}.pth",
                ),
                model,
                epoch,
                raw_top1,
            )

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"[EARLY STOP] no RAW Top-1 improvement larger than "
                f"{args.early_stop_min_delta:.6f} for "
                f"{args.early_stop_patience} epochs."
            )
            break

    training_loop_wall_seconds = time.perf_counter() - training_loop_start
    total_script_wall_seconds = time.perf_counter() - script_start_time
    summary = {
        "model": "AFGM_texture_priority",
        "ablate": args.ablate.upper(),
        "active_branches": used,
        "active_tag": used_tag,
        "device": str(DEVICE),
        "device_name": (
            torch.cuda.get_device_name(DEVICE)
            if DEVICE.type == "cuda"
            else platform.processor() or "CPU"
        ),
        "batch_size": int(args.batch_size),
        "configured_epochs": int(args.epoch),
        "completed_epochs": int(completed_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "best_epoch": int(best_epoch),
        "best_raw_top1": float(best_raw_top1),
        "total_train_seconds": float(total_train_seconds),
        "total_validation_seconds": float(total_validation_seconds),
        "training_loop_wall_seconds": float(training_loop_wall_seconds),
        "total_script_wall_seconds": float(total_script_wall_seconds),
        "average_train_seconds_per_epoch": float(
            total_train_seconds / max(1, completed_epochs)
        ),
        "average_validation_seconds_per_epoch": float(
            total_validation_seconds / max(1, completed_epochs)
        ),
        "time_to_best_wall_seconds": float(time_to_best_wall_seconds),
        "peak_train_cuda_memory_MB": float(
            max_peak_train_memory_mb
            if DEVICE.type == "cuda"
            else float("nan")
        ),
        "save_all_checkpoints": bool(args.save_ckpt),
    }
    with open(
        os.path.join(save_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_attention_checkpoint(
        os.path.join(save_dir, "last.pth"),
        model,
        max(0, completed_epochs - 1),
        best_raw_top1,
    )

    print(
        "[DONE] "
        f"best_raw_top1={best_raw_top1:.4f}, "
        f"best_epoch={best_epoch}, "
        f"completed_epochs={completed_epochs}, "
        f"total_train={total_train_seconds:.1f}s, "
        f"training_loop_wall={training_loop_wall_seconds:.1f}s, "
        f"peak_train_mem={summary['peak_train_cuda_memory_MB']:.1f}MB"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AFGM training aligned with the submitted manuscript."
    )

    parser.add_argument("--root_shape", default="data/CUB/feature_images/shape")
    parser.add_argument("--root_texture", default="data/CUB/feature_images/texture")
    parser.add_argument("--root_color", default="data/CUB/feature_images/color")

    parser.add_argument("--shape_model", default="checkpoints/hve/shape_resnet18.pth")
    parser.add_argument("--texture_model", default="checkpoints/hve/texture_resnet18.pth")
    parser.add_argument("--color_model", default="checkpoints/hve/color_resnet18.pth")

    parser.add_argument("--save_model_dir", default="outputs/AFGM")

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1.5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument(
        "--lr_drop_epoch",
        type=int,
        default=6,
        help="Begin the low-learning-rate refinement stage at this epoch.",
    )
    parser.add_argument(
        "--stage2_lr_scale",
        type=float,
        default=0.15,
        help="LR scale at the beginning of stage 2.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.05,
        help="Minimum LR ratio at the end of training.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without a meaningful Top-1 improvement; 0 disables.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0002,
        help="Minimum absolute Top-1 increase counted as an improvement.",
    )

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--multi_layer_head",
        action="store_true",
        help="Use the optional two-layer classifier; default is a linear head.",
    )
    parser.add_argument(
        "--gate_lr_scale",
        type=float,
        default=0.02,
        help="Gate-network LR divided by classifier LR.",
    )
    parser.add_argument(
        "--gate_warmup_epochs",
        type=int,
        default=0,
        help="0 means the gates learn from the first epoch.",
    )
    parser.add_argument("--gate_hidden", type=int, default=128)
    parser.add_argument("--gate_dropout", type=float, default=0.05)
    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--gate_min", type=float, default=0.85)
    parser.add_argument("--gate_max", type=float, default=1.15)

    parser.add_argument("--aux_weight", type=float, default=0.02)
    parser.add_argument("--shape_margin", type=float, default=0.01)
    parser.add_argument("--tc_tolerance", type=float, default=0.10)
    parser.add_argument("--lam_shape", type=float, default=1) 
    parser.add_argument("--lam_tc", type=float, default=1.0)
    parser.add_argument(
        "--gate_budget_weight",
        type=float,
        default=0.0,
        help="Optional penalty preventing all gates drifting upward together.",
    )
    parser.add_argument(
        "--gate_budget_target",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--texture_enhance",
        dest="texture_enhance",
        action="store_true",
        help="Enable the fixed negative-shape/positive-texture prior.",
    )
    parser.add_argument(
        "--no_texture_enhance",
        dest="texture_enhance",
        action="store_false",
        help="Disable the fixed biological prior.",
    )
    parser.set_defaults(texture_enhance=True)
    parser.add_argument(
        "--texture_boost_factor",
        type=float,
        default=1.0,  #1
    )
    parser.add_argument(
        "--phased_training",
        dest="phased_training",
        action="store_true",
        help="Gradually introduce gate regularization.",
    )
    parser.add_argument(
        "--no_phased_training",
        dest="phased_training",
        action="store_false",
        help="Use the full gate regularization from epoch 0.",
    )
    parser.set_defaults(phased_training=True)

    parser.add_argument(
        "--color_feat_dropout",
        type=float,
        default=0.0,  #0.0
    )
    parser.add_argument(
        "--branch_drop_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--branch_drop_s",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--branch_drop_t",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--branch_drop_c",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--feature_l2_norm",
        action="store_true",
    )

    parser.add_argument("--ema", action="store_true")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
    )
    parser.add_argument("--tta", action="store_true")

    parser.add_argument(
        "--ablate",
        default="none",
        choices=["none", "S", "T", "C"],
        help=(
            "One-click default removes T and retrains S+C. "
            "Change only this default to S/C/none for the other conditions."
        ),
    )
    parser.add_argument("--resume", default=None)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--save_ckpt",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save model_ck_0.pth ... model_ck_N.pth by default.",
        )
    else:
        parser.add_argument("--save_ckpt", action="store_true", default=True)

    return parser


if __name__ == "__main__":
    main()
