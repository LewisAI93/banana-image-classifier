from args import get_args
import pandas as pd
import os
import torch
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model
from augmentations import build_train_transforms, build_val_transforms


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

    print("=== Starting object detection training ===")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs (max): {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.wd}")
    print("=========================================\n")

    # 1. Read the dataframes
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df, transform=build_train_transforms(args.image_size))
    val_dataset = ObjDetectionDataset(val_df, transform=build_val_transforms(args.image_size))

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, 
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  collate_fn=collate, 
                            num_workers=0, pin_memory=torch.cuda.is_available())
    
    # 4. Initialising the model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)
    print("Model built. Starting training...\n")

    # 5. Train the model
    train_model(model, train_loader, val_loader, device)
    print("\nTraining finished.")


if __name__ == '__main__':
    main()