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
BATCH_SIZE = 8                    # increase if your GPU can handle it
NUM_WORKERS = 4                   # set to 0 on Windows or if issues
PIN_MEMORY = True
PERSISTENT_WORKERS = True

EPOCHS = 200                       # set total target epochs (increase if resuming)
LR = 1e-4
WARMUP_EPOCHS = 3                 # linear warmup
VAL_EVERY = 1                     # validate every `VAL_EVERY` epochs; set to >1 to skip often
FREEZE_BACKBONE_EPOCHS = 10       # freeze backbone for these many epochs (0 to disable)
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ------------------------------------------

def move_targets_to_device(targets, device):
    """Move list[dict] targets to device. Keeps non-tensor values untouched."""
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                new_t[k] = v.to(device)
            elif isinstance(v, (list, tuple)):
                # convert list of tensors to tensors on device if possible
                new_list = []
                for item in v:
                    if torch.is_tensor(item):
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                new_t[k] = new_list
            else:
                new_t[k] = v
        new_targets.append(new_t)
    return new_targets

def save_checkpoint(path, model, optimizer, scheduler, epoch):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict() if optimizer is not None else None,
        'sched_state': scheduler.state_dict() if scheduler is not None else None
    }, path)

def main():
    logger = get_logger('training')
    logger.print_banner()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    use_amp = (device.type == "cuda")

    # Data loaders
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

    num_classes = 26
    model = DETR(num_classes=num_classes)
    model.to(device)

    # --------- RESUME TRAINING ----------
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, 'final_model.pt')
    start_epoch = 0
    checkpoint = None
    if os.path.exists(RESUME_PATH):
        logger.info(f"Resuming training from checkpoint: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        # load model weights
        try:
            model.load_state_dict(checkpoint['model_state'])
            start_epoch = int(checkpoint.get('epoch', 0))
            logger.info(f"Checkpoint loaded. Starting at epoch {start_epoch}.")
        except Exception as e:
            logger.warning(f"Could not load model state from checkpoint: {e}")
    else:
        logger.info('No checkpoint found. Starting training from scratch.')

    # If resuming beyond total epochs, warn and exit
    if start_epoch >= EPOCHS:
        logger.warning(f"Checkpoint epoch ({start_epoch}) >= target EPOCHS ({EPOCHS}). Nothing to do.")
        return

    # If resuming past freeze point, ensure backbone is unfrozen
    if start_epoch >= FREEZE_BACKBONE_EPOCHS and FREEZE_BACKBONE_EPOCHS > 0 and hasattr(model, 'backbone'):
        logger.info("Resuming past freeze epoch -> ensuring backbone parameters are unfrozen.")
        for p in model.backbone.parameters():
            p.requires_grad = True

    # Optionally freeze backbone initially to speed up (only if starting fresh or start_epoch < FREEZE_BACKBONE_EPOCHS)
    if FREEZE_BACKBONE_EPOCHS > 0 and hasattr(model, 'backbone') and start_epoch < FREEZE_BACKBONE_EPOCHS:
        logger.info(f"Backbone will be frozen for first {FREEZE_BACKBONE_EPOCHS} epochs (resuming from {start_epoch}).")
        for p in model.backbone.parameters():
            p.requires_grad = False

    # create optimizer after possibly freezing params
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # restore optimizer state if checkpoint exists and shapes match
    if checkpoint is not None and checkpoint.get('opt_state') is not None:
        try:
            ck_opt_state = checkpoint['opt_state']
            # quick sanity check: param_group counts should match
            if 'param_groups' in ck_opt_state and len(ck_opt_state['param_groups']) == len(optimizer.state_dict()['param_groups']):
                optimizer.load_state_dict(ck_opt_state)
                logger.info("Optimizer state restored.")
            else:
                logger.warning("Optimizer param_groups mismatch. Skipping optimizer state restore.")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    # Warmup (linear) for WARMUP_EPOCHS, then step decay at 70% and 90%
    total_epochs = EPOCHS
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=WARMUP_EPOCHS),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.7 * total_epochs), int(0.9 * total_epochs)],
                gamma=0.1
            )
        ],
        milestones=[WARMUP_EPOCHS]
    )

    # restore scheduler state if possible
    if checkpoint is not None and checkpoint.get('sched_state') is not None:
        try:
            scheduler.load_state_dict(checkpoint['sched_state'])
            logger.info('Scheduler state restored.')
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")

    weights = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    criterion = criterion.to(device)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    train_batches = len(train_loader)
    test_batches = len(test_loader)

    logger.info(f"Train batches: {train_batches}, Test batches: {test_batches}, Batch size: {BATCH_SIZE}")

    for epoch in range(start_epoch, EPOCHS):  # resume from start_epoch
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=train_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", ncols=120)

        for batch_idx, batch in pbar:
            images, targets = batch  # stacker must return (images_tensor, list_of_targets)
            # Sanity: move images & targets
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets)
                    w = criterion.weight_dict
                    losses = (
                        loss_dict['labels']['loss_ce'] * w['class_weighting'] +
                        loss_dict['boxes']['loss_bbox'] * w['bbox_weighting'] +
                        loss_dict['boxes']['loss_giou'] * w['giou_weighting']
                    )

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += losses.item()
                running_avg = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': f"{running_avg:.5f}"
                })

            except Exception as e:
                logger.error(f"Training error epoch {epoch} batch {batch_idx}: {e}")
                logger.error(f"Batch target preview: {targets[0] if len(targets)>0 else 'empty'}")
                continue

        pbar.close()

        # If we reach the freeze -> unfreeze transition and we started before it, do it now
        if FREEZE_BACKBONE_EPOCHS > 0 and start_epoch < FREEZE_BACKBONE_EPOCHS and epoch + 1 == FREEZE_BACKBONE_EPOCHS:
            if hasattr(model, 'backbone'):
                logger.info("Unfreezing backbone parameters")
                for p in model.backbone.parameters():
                    p.requires_grad = True
                # recreate optimizer and scheduler after unfreezing
                optimizer = optim.Adam(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=WARMUP_EPOCHS),
                        torch.optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(0.7 * total_epochs), int(0.9 * total_epochs)],
                            gamma=0.1
                        )
                    ],
                    milestones=[WARMUP_EPOCHS]
                )
                scaler = torch.amp.GradScaler(enabled=use_amp)

        # Validation (run every VAL_EVERY epochs)
        val_loss = None
        if (epoch % VAL_EVERY) == 0:
            model.eval()
            test_running = 0.0
            vbar = tqdm(enumerate(test_loader), total=test_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", ncols=120)
            with torch.no_grad():
                for v_idx, vbatch in vbar:
                    v_images, v_targets = vbatch
                    v_images = v_images.to(device, non_blocking=True)
                    v_targets = move_targets_to_device(v_targets, device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        v_outputs = model(v_images)
                        v_loss_dict = criterion(v_outputs, v_targets)
                        w = criterion.weight_dict
                        v_losses = (
                            v_loss_dict['labels']['loss_ce'] * w['class_weighting'] +
                            v_loss_dict['boxes']['loss_bbox'] * w['bbox_weighting'] +
                            v_loss_dict['boxes']['loss_giou'] * w['giou_weighting']
                        )

                    test_running += v_losses.item()
                    running_val_avg = test_running / (v_idx + 1)
                    vbar.set_postfix({'val_loss': f"{running_val_avg:.5f}"})

            vbar.close()
            val_loss = test_running / max(1, test_batches)

        # step scheduler once per epoch (after validation)
        scheduler.step()

        # Save checkpoint every 10 epochs and final
        if (epoch + 1) % 10 == 0:
            cp = os.path.join(CHECKPOINT_DIR, f"{epoch+1}_model.pt")
            save_checkpoint(cp, model, optimizer, scheduler, epoch+1)
            logger.info(f"Saved checkpoint: {cp}")

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.1f}s - train_loss={running_loss/train_batches:.5f}"
                    + (f", val_loss={val_loss:.5f}" if val_loss is not None else ""))

    # final save
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, EPOCHS)
    logger.info(f"Training finished. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
