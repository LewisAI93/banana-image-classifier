from args import get_args
import pandas as pd
import os
import torch
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
    
def main():
    args = get_args()
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')


    # 1. Read the dataframes
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, 
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  collate_fn=collate, 
                            num_workers=0, pin_memory=torch.cuda.is_available())
    
    # 4. Initialising the model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)

    # 5. Train the model
    train_model(model, train_loader, val_loader, device)

    print()


if __name__ == '__main__':
    main()