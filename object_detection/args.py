import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet50_fpn',
                        choices=['fasterrcnn_resnet50_fpn','fasterrcnn_mobilenet_v3'])

    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
   
    
    parser.add_argument('--csv_dir', type=str, default='object_detection/data/CSVs')
    parser.add_argument('--out_dir', type=str, default='./sessions')

    parser.add_argument('--batch_size', type=int, default=8, choices=[8,16,32,64])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001) # learning rate, adjust for learning curve
    parser.add_argument('--wd', type=float, default=1e-4) # weight decay. Other options: 1e-5 or 1e-7

    return parser.parse_args()