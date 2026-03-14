from args import get_args
import pandas as pd
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import os
from model import build_model


def collate(batch):
    images, targets = zip(batch)
    return list(images), list(targets)

def main():
    args = get_args()

    # 1. Read the dataframes

    train_df = pd.read_csv(os.path.join(args.csv_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "val_df.csv"))

    # 2. Prepare datasets

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create data loaders

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 4. Initializing the model
    
    model = build_model(args.backbone)


    print()

if __name__ == "__main__":
    main()