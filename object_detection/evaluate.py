import os
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from args import get_args
from model import build_model
from augmentations import build_val_transforms


@torch.no_grad()
def predict_single_image(model, device, img_path, val_transforms, thresh):
    pil_img = Image.open(img_path).convert("RGB")
    img = pil_img
    target = None
    for t in val_transforms:
        img, target = t(img, target)
    img = img.to(device)
    outputs = model([img])[0]
    scores = outputs["scores"]
    keep = scores >= thresh
    outputs = {k: v[keep] for k, v in outputs.items()}
    return pil_img, img, outputs



@torch.no_grad()
def run_inference_on_test_images():
    args = get_args()
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "sessions" / "best_model.pth"
    image_dir = base_dir / "data" / "test_images"

    model = build_model(args.backbone, num_classes=args.num_classes + 1)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_transforms = build_val_transforms(args.image_size)
    thresh = 0.5

    for name in os.listdir(image_dir):
        img_path = image_dir / name
        if not img_path.is_file():
            continue

        pil_img, img, outputs = predict_single_image(model, device, img_path, val_transforms, thresh)
        boxes = outputs["boxes"]
        scores = outputs["scores"]

        print(f"{name}: {len(boxes)} detections above {thresh}")

        if boxes.numel() == 0:
            pil_img.show()
            input("No detections. Press Enter for next image...")
            continue

        top_idx = scores.argmax()
        boxes = boxes[top_idx:top_idx+1]
        scores = scores[top_idx:top_idx+1]

        img_uint8 = (img.cpu().clamp(0, 1) * 255).to(torch.uint8)
        class_name = "banana"
        label_strings = [f"{class_name} {s:.2f}" for s in scores.cpu().tolist()]

        result = draw_bounding_boxes(
            img_uint8,
            boxes.cpu(),
            labels=label_strings,
            colors="yellow",
            width=2,
        )

        out_pil = F.to_pil_image(result)
        out_pil.show()
        input("Press Enter for next image...")


if __name__ == "__main__":
    run_inference_on_test_images()