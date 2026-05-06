import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
from dataset import get_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = '/home/vteam2/ayesha_xbd/preprocessed'
SAVE_DIR         = '/home/vteam2/ayesha_xbd/segFinal/checkpoints/segformer_b1'
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES      = 4
IMAGE_SIZE       = 256
BATCH_SIZE       = 8
EPOCHS           = 50
LR               = 6e-5
GRAD_ACCUM_STEPS = 2
NUM_WORKERS      = 4
PATIENCE         = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── ImageNet normalization ────────────────────────────────────────────────────
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(DEVICE)

def normalize(x):
    return (x - MEAN) / STD

# ── Dataloaders ───────────────────────────────────────────────────────────────
print("\nLoading data...")
train_loader, val_loader, test_loader = get_dataloaders(
    PREPROCESSED_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nSanity check on first batch:")
for pre, post, mask in train_loader:
    print(f"  Post shape  : {post.shape}")
    print(f"  Mask shape  : {mask.shape}")
    print(f"  Mask values : {mask.unique().tolist()}")
    print(f"  Post range  : [{post.min():.2f}, {post.max():.2f}]")
    break

# ── Model ─────────────────────────────────────────────────────────────────────
print("\nLoading SegFormer-B1...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "/home/vteam2/ayesha_xbd/segFormerOriginal",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ── Class weights ─────────────────────────────────────────────────────────────
def calculate_class_weights(loader, num_classes=NUM_CLASSES, num_batches=50):
    print("\nCalculating class weights...")
    counts = torch.zeros(num_classes)

    for i, (_, _, masks) in enumerate(loader):
        if i >= num_batches:
            break
        for c in range(num_classes):
            counts[c] += (masks == c).sum()
    counts  = counts + 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    print("Class weights:")
    for name, w in zip(class_names, weights):
        print(f"  {name:>15}: {w:.4f}")
    return weights.to(DEVICE)

class_weights = calculate_class_weights(train_loader)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
encoder_params = list(model.segformer.parameters())
decoder_params = list(model.decode_head.parameters())

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': LR * 0.1},
    {'params': decoder_params, 'lr': LR}
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)
scaler = GradScaler()

# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics(logits_up, masks, num_classes=NUM_CLASSES):
    preds = logits_up.argmax(dim=1)
    preds = preds.cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()

    pixel_acc     = (preds == masks).mean()
    iou_per_class = []
    f1_per_class  = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (masks == c)

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
        'f1_per_class':  f1_per_class,
    }

# ── Training loop ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
best_miou   = 0.0
no_improve  = 0

print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...\n")

for epoch in range(EPOCHS):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss    = 0.0
    train_metrics = {'pixel_acc': 0, 'miou': 0, 'mf1': 0}
    optimizer.zero_grad()

    for step, (pre, post, mask) in enumerate(train_loader):
        post = normalize(post.to(DEVICE))
        mask = mask.to(DEVICE)

        with autocast():
            outputs   = model(pixel_values=post)
            logits    = outputs.logits
            logits_up = F.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear',
                align_corners=False
            )
            loss = criterion(logits_up, mask) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * GRAD_ACCUM_STEPS

        metrics = calculate_metrics(logits_up.detach(), mask.detach())
        for k in train_metrics:
            train_metrics[k] += metrics[k]

    n_train           = len(train_loader)
    avg_train_loss    = train_loss / n_train
    avg_train_metrics = {k: v / n_train for k, v in train_metrics.items()}

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    val_loss          = 0.0
    val_metrics       = {'pixel_acc': 0, 'miou': 0, 'mf1': 0}
    val_iou_per_class = np.zeros(NUM_CLASSES)
    val_f1_per_class  = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for pre, post, mask in val_loader:
            post = normalize(post.to(DEVICE))
            mask = mask.to(DEVICE)

            with autocast():
                outputs   = model(pixel_values=post)
                logits    = outputs.logits
                logits_up = F.interpolate(
                    logits,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode='bilinear',
                    align_corners=False
                )
                loss = criterion(logits_up, mask)

            val_loss += loss.item()
            metrics   = calculate_metrics(logits_up, mask)

            for k in val_metrics:
                val_metrics[k] += metrics[k]
            val_iou_per_class += np.array(metrics['iou_per_class'])
            val_f1_per_class  += np.array(metrics['f1_per_class'])

    n_val           = len(val_loader)
    avg_val_loss    = val_loss / n_val
    avg_val_metrics = {k: v / n_val for k, v in val_metrics.items()}
    avg_val_iou     = val_iou_per_class / n_val
    avg_val_f1      = val_f1_per_class  / n_val

    scheduler.step()

    # ── Print epoch summary ────────────────────────────────────────────────────
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Train mIoU: {avg_train_metrics['miou']:.4f} | Val mIoU: {avg_val_metrics['miou']:.4f}")
    print(f"  Train F1  : {avg_train_metrics['mf1']:.4f}  | Val F1  : {avg_val_metrics['mf1']:.4f}")
    print(f"  Pixel Acc : {avg_val_metrics['pixel_acc']:.4f}")
    print(f"  Per-class Val IoU & F1:")
    for name, iou, f1 in zip(CLASS_NAMES, avg_val_iou, avg_val_f1):
        print(f"    {name:>15}: IoU={iou:.4f}  F1={f1:.4f}")

    # ── Early stopping & save ──────────────────────────────────────────────────
    if avg_val_metrics['miou'] > best_miou:
        best_miou  = avg_val_metrics['miou']
        no_improve = 0
        torch.save({
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_miou':             best_miou,
            'val_loss':             avg_val_loss,
        }, os.path.join(SAVE_DIR, 'best_segformer_b1.pth'))
        print(f"  ✅ Saved best model (mIoU={best_miou:.4f})")
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve}/{PATIENCE} epochs")
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

# ── Save final model ──────────────────────────────────────────────────────────
torch.save({
    'epoch':                epoch + 1,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_miou':             avg_val_metrics['miou'],
    'val_loss':             avg_val_loss,
}, os.path.join(SAVE_DIR, 'final_segformer_b1.pth'))

print(f"\nTraining complete. Best Val mIoU: {best_miou:.4f}")
print(f"Models saved to: {SAVE_DIR}")