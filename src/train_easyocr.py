import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

try:
    from tqdm import tqdm
except ImportError:
    class tqdm_fallback:
        def __init__(self, iterable, desc=''):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
        def __iter__(self):
            print(f"Starting: {self.desc}")
            for i, item in enumerate(self.iterable):
                if self.total and (i == 0 or (i + 1) % max(1, self.total // 10) == 0 or i == self.total - 1):
                    print(f"{self.desc} Progress: {i+1}/{self.total}")
                yield item
        def set_postfix(self, *args, **kwargs):
            pass
        def set_description(self, *args, **kwargs):
            pass
    def tqdm(iterable, *args, **kwargs):
        desc = kwargs.get('desc', '')
        return tqdm_fallback(iterable, desc=desc)

# ==========================================
# 1. DATA AUGMENTATION PIPELINE
# ==========================================

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
    train_transforms = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(50, 100), p=0.3),
        A.OpticalDistortion(distort_limit=0.1, p=0.2),
        A.InvertImg(p=0.1),
    ])
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    train_transforms = None

def apply_augmentations(img, train_mode=True):
    """
    Applies high-quality data augmentations to help model generalize
    to different lighting, blur, noise, and angle variations.
    """
    if not train_mode:
        return img
    
    if ALBUMENTATIONS_AVAILABLE and train_transforms is not None:
        res = train_transforms(image=img)
        return res['image']
    
    # Fallback CV2 augmentations
    if np.random.rand() < 0.3:
        angle = np.random.uniform(-4, 4)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
    if np.random.rand() < 0.4:
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, np.random.uniform(1, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    if np.random.rand() < 0.15:
        kernel = np.ones((2, 2), np.uint8)
        if np.random.rand() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)
            
    if np.random.rand() < 0.1:
        img = cv2.bitwise_not(img)
            
    return img

# ==========================================
# 2. PYTORCH DATASET AND COLLATOR
# ==========================================

class EasyOCRDataset(Dataset):
    def __init__(self, txt_path, data_dir, img_w=256, img_h=64, train_mode=True):
        self.data_dir = data_dir
        self.img_w = img_w
        self.img_h = img_h
        self.train_mode = train_mode
        
        self.samples = []
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Annotation file not found: {txt_path}")
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))
                    
        print(f"Loaded {len(self.samples)} samples from {txt_path} (train_mode={train_mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, rel_path)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Safe read for paths with Cyrillic characters on Windows
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {img_path}")
            
        # Apply augmentations
        img = apply_augmentations(img, train_mode=self.train_mode)
        
        # Resize to fixed width & height (EasyOCR generation2 uses height 64)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize and add channel dimension
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        
        return torch.tensor(img), label

def collate_fn(batch, converter):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Encode label strings using EasyOCR's CTCLabelConverter
    text_tensor, length_tensor = converter.encode(labels)
    
    return images, text_tensor, length_tensor

# ==========================================
# 3. METRIC ESTIMATIONS
# ==========================================

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def evaluate(model, dataloader, converter, device):
    model.eval()
    correct = 0
    total = 0
    total_cer = 0.0
    total_chars = 0
    
    with torch.no_grad():
        for images, target_tensor, target_lengths in dataloader:
            images = images.to(device)
            preds = model(images, None)  # [batch_size, 63, 97]
            
            _, max_indices = preds.max(2)
            batch_size = images.size(0)
            seq_len = max_indices.size(1)
            
            flat_preds = max_indices.flatten().cpu().numpy()
            pred_lengths = np.array([seq_len] * batch_size, dtype=np.int32)
            
            pred_strings = converter.decode_greedy(flat_preds, pred_lengths)
            
            # Decode target label strings
            target_flat = target_tensor.cpu().numpy()
            target_lens_np = target_lengths.cpu().numpy()
            target_strings = converter.decode_greedy(target_flat, target_lens_np)
            
            for pred_str, target_str in zip(pred_strings, target_strings):
                pred_str = pred_str.strip()
                target_str = target_str.strip()
                
                dist = edit_distance(pred_str, target_str)
                total_cer += dist
                total_chars += len(target_str) if len(target_str) > 0 else 1
                
                if pred_str == target_str:
                    correct += 1
                total += 1
                
    accuracy = correct / total if total > 0 else 0.0
    cer = total_cer / total_chars if total_chars > 0 else 0.0
    return accuracy, cer

# ==========================================
# 4. TRAINING MASTER LOOP
# ==========================================

def train():
    parser = argparse.ArgumentParser(description="EasyOCR Fine-Tuning Script")
    parser.add_argument("--selected-dir", default="selected", help="Path to selected/ frames folder")
    parser.add_argument("--manifest", default="selected_manifest.csv", help="Path to manifest file")
    parser.add_argument("--yolo-model", default="detector_for_annotation.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--csv", default="results12.csv", help="Path to audited CSV results")
    parser.add_argument("--data-dir", default="train_ocr_data", help="OCR crops directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--save-model", default="best_easyocr.pth", help="Path to save the best model weights")
    parser.add_argument("--pretrained-weights", default="", help="Path to pretrained english_g2.pth")
    parser.add_argument("--resume-model", default="", help="Resume training from a saved train_easyocr checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    # Step 1: Check if dataset needs to be prepared
    train_txt = os.path.join(args.data_dir, "train.txt")
    val_txt = os.path.join(args.data_dir, "val.txt")
    
    if not (os.path.exists(train_txt) and os.path.exists(val_txt)):
        print("\nOCR dataset splits not found. Preparing dataset first...")
        import subprocess
        prep_script = "prepare_ocr_dataset_v2.py"
        if not os.path.exists(prep_script):
            prep_script = os.path.join(os.path.dirname(__file__), "prepare_ocr_dataset_v2.py")
            
        if not os.path.exists(prep_script):
            raise FileNotFoundError("Could not find prepare_ocr_dataset_v2.py!")
            
        import sys
        cmd = [
            sys.executable, prep_script,
            "--selected-dir", args.selected_dir,
            "--manifest", args.manifest,
            "--out-dir", args.data_dir
        ]
        
        if os.path.exists(args.csv):
            cmd += ["--csv-results", args.csv]
            print(f"Using audited CSV: {args.csv}")
        elif os.path.exists(args.yolo_model):
            cmd += ["--model", args.yolo_model]
            print(f"Using YOLO model for online generation: {args.yolo_model}")
        else:
            fallback_csv = "local_results_full.csv"
            if os.path.exists(fallback_csv):
                cmd += ["--csv-results", fallback_csv]
                print(f"Using fallback CSV: {fallback_csv}")
            else:
                raise FileNotFoundError("Either results CSV or YOLO weights are required for dataset generation!")
                
        print(f"Running dataset prep: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Dataset prepared successfully!\n")

    # Step 2: Initialize EasyOCR configuration and converter
    print("Loading EasyOCR components...")
    try:
        import easyocr
        from easyocr.config import recognition_models
        from easyocr.utils import CTCLabelConverter
        from easyocr.model.vgg_model import Model
        
        # Суженный алфавит. Оставляем все заглавные буквы (для дней недели MON, TUE) и цифры.
        # Строчные буквы убираем, чтобы не было путаницы (например l и 1).
        # Модель всё равно выучит нуль с точкой как "0", так как в Ground Truth разметке будут нули.
        characters = "0123456789-/:. ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    except ImportError:
        raise ImportError("easyocr library is not installed. Please run 'pip install easyocr'")

    converter = CTCLabelConverter(characters)
    num_class = len(converter.character)  # should be 97 (96 chars + 1 blank)
    print(f"Character dictionary initialized with {len(characters)} characters. Total classes: {num_class}")

    # Step 3: Instantiate VGG recognition model
    model = Model(input_channel=1, output_channel=256, hidden_size=256, num_class=num_class).to(device)

    resume_checkpoint = None
    start_epoch = 1
    best_accuracy = 0.0

    # Step 4: Resume checkpoint or load pretrained english_g2 weights
    if args.resume_model:
        if not os.path.exists(args.resume_model):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_model}")
        print(f"Resuming training from checkpoint: {args.resume_model}")
        resume_checkpoint = torch.load(args.resume_model, map_location=device)
        if "model_state_dict" not in resume_checkpoint:
            raise ValueError("--resume-model must point to a train_easyocr checkpoint with model_state_dict")
        ckpt_characters = resume_checkpoint.get("characters", characters)
        if ckpt_characters != characters:
            print("Warning: checkpoint character set differs from current character set.")
            print(f"Checkpoint: {ckpt_characters}")
            print(f"Current:    {characters}")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_accuracy = float(resume_checkpoint.get("best_val_accuracy", resume_checkpoint.get("val_accuracy", 0.0)))
        print(f"Loaded epoch {start_epoch - 1}. Best validation accuracy so far: {best_accuracy*100:.2f}%")

    if not args.resume_model and not args.pretrained_weights:
        # Check standard user dir
        default_path = os.path.expanduser('~/.EasyOCR/model/english_g2.pth')
        if os.path.exists(default_path):
            args.pretrained_weights = default_path
        else:
            # Check current dir
            if os.path.exists("english_g2.pth"):
                args.pretrained_weights = "english_g2.pth"
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "english_g2.pth")):
                args.pretrained_weights = os.path.join(os.path.dirname(__file__), "english_g2.pth")

    if not args.resume_model and args.pretrained_weights and os.path.exists(args.pretrained_weights):
        print(f"Loading pretrained recognition weights from: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        try:
            model.load_state_dict(new_state_dict)
            print("Successfully loaded pre-trained weights.")
        except Exception as e:
            print(f"Error loading full state dict: {e}")
            print("Attempting to load weights partially...")
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(matched_dict)} / {len(model_dict)} weight layers.")
    elif not args.resume_model:
        print("Warning: Pretrained weights file english_g2.pth not found. Model will train from scratch!")

    # Step 5: Datasets and Loaders
    train_dataset = EasyOCRDataset(train_txt, args.data_dir, train_mode=True)
    val_dataset = EasyOCRDataset(val_txt, args.data_dir, train_mode=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, converter), num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, converter), num_workers=0
    )

    # Step 6: Loss, Optimizer and Scheduler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    remaining_epochs = max(1, args.epochs - start_epoch + 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=len(train_loader), epochs=remaining_epochs,
        pct_start=0.1 # 10% времени на warmup
    )

    if resume_checkpoint and "optimizer_state_dict" in resume_checkpoint:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as e:
            print(f"Could not load optimizer state, continuing with a fresh optimizer: {e}")

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}, target epochs={args.epochs}. Nothing to train.")
        return

    print(f"\nTraining started. Train set={len(train_dataset)}, Val set={len(val_dataset)}, Epochs={start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, target_tensor, target_lengths in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()
            preds = model(images, None)  # [batch_size, 63, num_class]
            
            preds_ctc = preds.permute(1, 0, 2)
            preds_log_softmax = preds_ctc.log_softmax(2)
            
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=preds_ctc.size(0),
                dtype=torch.long, device=device
            )
            
            loss = criterion(preds_log_softmax, target_tensor.to(device), input_lengths, target_lengths.to(device))
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step() # Шаг шедулера на каждый батч для OneCycleLR
                epoch_loss += loss.item() * images.size(0)
            
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_dataset)

        # Evaluate on validation split
        val_acc, val_cer = evaluate(model, val_loader, converter, device)
        
        print(f"Summary Epoch {epoch}/{args.epochs} | Avg Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% | Val CER: {val_cer*100:.2f}%")
        
        if val_acc > best_accuracy or epoch == args.epochs:
            best_accuracy = max(val_acc, best_accuracy)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'characters': characters,
                'num_class': num_class,
                'epoch': epoch,
                'val_accuracy': val_acc,
                'best_val_accuracy': best_accuracy
            }, args.save_model)
            print(f"--> Saved best model checkpoint to '{args.save_model}' with validation accuracy: {val_acc*100:.2f}%")

    print(f"\nEasyOCR Training complete! Best validation accuracy: {best_accuracy*100:.2f}%")

if __name__ == "__main__":
    train()
