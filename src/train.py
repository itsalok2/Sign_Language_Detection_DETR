import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim

from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from utils.boxes import stacker
from utils.logger import get_logger
from torch import save

# ----------------- CONFIG -----------------
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

EPOCHS = 300
WARMUP_STEPS = 1000          # FIX: this is steps, not epochs
VAL_EVERY = 1

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ------------------------------------------


def move_targets_to_device(targets, device):
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                new_t[k] = v.to(device)
            else:
                new_t[k] = v
        new_targets.append(new_t)
    return new_targets


def save_checkpoint(path, model, optimizer, scheduler, epoch):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict()
    }, path)


def main():
    logger = get_logger('training')
    logger.print_banner()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    use_amp = (device.type == "cuda")

    # ---------------- DATA ----------------
    train_dataset = DETRData('data/train', train=True)
    test_dataset = DETRData('data/test', train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=stacker,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=stacker,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )

    # ---------------- MODEL ----------------
    num_classes = 26
    model = DETR(num_classes=num_classes).to(device)

    # ---------------- OPTIMIZER ----------------
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.transformer.parameters(), 'lr': 1e-4},
        {'params': model.linear_class.parameters(), 'lr': 1e-4},
        {'params': model.linear_bbox.parameters(), 'lr': 1e-4},
        {'params': model.query_pos, 'lr': 1e-4},
    ], weight_decay=1e-4)

    total_steps = EPOCHS * len(train_loader)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=WARMUP_STEPS
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.7 * total_steps), int(0.9 * total_steps)],
                gamma=0.1
            )
        ],
        milestones=[WARMUP_STEPS]
    )

    # ---------------- LOSS ----------------
    weights = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weights,
        eos_coef=0.05          # FIX: better for many classes
    ).to(device)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---------------- TRAIN ----------------
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120)
        for batch_idx,(images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)

            with torch.amp.autocast(enabled=use_amp,device_type=device.type):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                w = criterion.weight_dict
                loss = (
                    loss_dict['labels']['loss_ce'] * w['class_weighting'] +
                    loss_dict['boxes']['loss_bbox'] * w['bbox_weighting'] +
                    loss_dict['boxes']['loss_giou'] * w['giou_weighting']
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() <= prev_scale:
                scheduler.step()

            running_loss += loss.item()
            global_step += 1
            avg_loss = running_loss / (batch_idx + 1)
            # logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

            pbar.set_postfix({
                "train_loss": f"{avg_loss:.4f}",
                "lr_bb": f"{optimizer.param_groups[0]['lr']:.1e}",
                "lr_tr": f"{optimizer.param_groups[1]['lr']:.1e}",
            })

        pbar.close()
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        # ---------------- VALIDATION ----------------
        if epoch % VAL_EVERY == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(device)
                    targets = move_targets_to_device(targets, device)

                    outputs = model(images)
                    loss_dict = criterion(outputs, targets)
                    w = criterion.weight_dict
                    loss = (
                        loss_dict['labels']['loss_ce'] * w['class_weighting'] +
                        loss_dict['boxes']['loss_bbox'] * w['bbox_weighting'] +
                        loss_dict['boxes']['loss_giou'] * w['giou_weighting']
                    )
                    val_loss += loss.item()

            val_loss /= max(1, len(test_loader))
            logger.info(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

        # ---------------- CHECKPOINT ----------------
        if (epoch + 1) % 10 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"{epoch+1}_model.pt")
            save_checkpoint(path, model, optimizer, scheduler, epoch + 1)
            logger.info(f"Saved checkpoint: {path}")

    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, EPOCHS)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
