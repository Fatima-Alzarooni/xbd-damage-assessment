# training script for SiameseSegFormer on xBD dataset
# same structure as train_unet_base.py but uses SegFormer backbone
# pretrained nvidia/mit-b2 with 6-channel input (pre + post concatenated)

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler
import sys
sys.path.insert(0, '/home/vteam2/ayesha_xbd')

from dataset import get_dataloaders
from my_segformer import SegFormer

# config
PREPROCESSED_DIR = '/home/vteam2/ayesha_xbd/preprocessed'
SAVE_DIR         = Path('/home/vteam2/ayesha_xbd/checkpoints/segformer_run1')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES   = 4
BATCH_SIZE    = 8
NUM_EPOCHS    = 30
LEARNING_RATE = 0.0001  # lower lr for pretrained model
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


# performance metrics
def calculate_metrics(preds, masks, num_classes=4):
    preds = torch.argmax(preds, dim=1)
    preds = preds.cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()

    # ignore unclassified pixels (value 4)
    valid = masks != 4
    preds = preds[valid]
    masks = masks[valid]

    pixel_acc     = (preds == masks).mean()
    iou_per_class = []
    f1_per_class  = []

    for c in range(num_classes):
        pred_c       = (preds == c)
        true_c       = (masks == c)
        intersection = (pred_c & true_c).sum()
        union        = (pred_c | true_c).sum()

        iou = intersection / (union + 1e-8)
        iou_per_class.append(iou)

        tp = intersection
        fp = (pred_c & ~true_c).sum()
        fn = (~pred_c & true_c).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_per_class.append(f1)

    return {
        'pixel_acc':     pixel_acc,
        'miou':          np.mean(iou_per_class),
        'mf1':           np.mean(f1_per_class),
        'iou_per_class': iou_per_class,
        'f1_per_class':  f1_per_class
    }


def calculate_confusion_matrix(preds, masks, num_classes=4):
    preds = torch.argmax(preds, dim=1).cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()

    valid = masks != 4
    preds = preds[valid]
    masks = masks[valid]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(masks, preds):
        cm[t][p] += 1
    return cm


# training loop
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss  = 0
    all_metrics = {'pixel_acc': 0, 'miou': 0, 'mf1': 0}

    for pre, post, masks in loader:
        pre   = pre.to(device)
        post  = post.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(pre, post)
            loss    = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics = calculate_metrics(outputs.detach(), masks.detach())
        for k in all_metrics:
            all_metrics[k] += metrics[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0
    all_metrics   = {'pixel_acc': 0, 'miou': 0, 'mf1': 0}
    iou_per_class = np.zeros(NUM_CLASSES)
    confusion_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for pre, post, masks in loader:
            pre   = pre.to(device)
            post  = post.to(device)
            masks = masks.to(device)

            with autocast('cuda'):
                outputs = model(pre, post)
                loss    = criterion(outputs, masks)

            total_loss += loss.item()
            metrics     = calculate_metrics(outputs, masks)

            for k in all_metrics:
                all_metrics[k] += metrics[k]

            iou_per_class += np.array(metrics['iou_per_class'])
            confusion_mat += calculate_confusion_matrix(outputs, masks)

    n = len(loader)
    avg_metrics = {k: v / n for k, v in all_metrics.items()}
    avg_metrics['iou_per_class'] = iou_per_class / n

    return total_loss / n, avg_metrics, confusion_mat


def main():
    print(f"Using device: {DEVICE}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        PREPROCESSED_DIR, batch_size=BATCH_SIZE, drop_no_damage=False
    )

    print("\nInitializing SegFormer...")
    model = SegFormer(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # aggressive class weights + ignore unclassified
    class_weights = torch.tensor([0.2, 3.0, 4.0, 4.0]).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, ignore_index=4)

    # lower lr for pretrained backbone
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler    = GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou': [],
        'train_f1':   [], 'val_f1':   []
    }

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_miou   = 0.0
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    for epoch in range(NUM_EPOCHS):

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler
        )

        val_loss, val_metrics, confusion_mat = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_miou'].append(float(train_metrics['miou']))
        history['val_miou'].append(float(val_metrics['miou']))
        history['train_f1'].append(float(train_metrics['mf1']))
        history['val_f1'].append(float(val_metrics['mf1']))

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train mIoU: {train_metrics['miou']:.4f} | Val mIoU: {val_metrics['miou']:.4f}")
        print(f"  Train F1:   {train_metrics['mf1']:.4f}  | Val F1:   {val_metrics['mf1']:.4f}")
        print(f"  Pixel Acc:  {val_metrics['pixel_acc']:.4f}")
        print("  Per class IoU:")
        for name, iou in zip(class_names, val_metrics['iou_per_class']):
            print(f"    {name}: {iou:.4f}")

        if (epoch + 1) % 10 == 0:
            print("\n  Confusion Matrix (rows=true, cols=predicted):")
            header = "  " + "".join(f"{n:>15}" for n in class_names)
            print(header)
            for i, row in enumerate(confusion_mat):
                print(f"  {class_names[i]:>15}" + "".join(f"{v:>15}" for v in row))

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou':             best_miou,
                'val_loss':             val_loss,
            }, SAVE_DIR / 'best_segformer_run1.pth')
            print(f"  Saved best model (mIoU: {best_miou:.4f})")

    torch.save({
        'epoch':                NUM_EPOCHS,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_miou':             val_metrics['miou'],
        'val_loss':             val_loss,
    }, SAVE_DIR / 'final_segformer_run1.pth')

    with open(SAVE_DIR / 'training_history_segformer_run1.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved!")

    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    print(f"Models saved to {SAVE_DIR}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()