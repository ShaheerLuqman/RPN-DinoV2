#!/usr/bin/env python3
"""Phase-1 Approach A: train a single-class ('object') detector for localization.

Class-agnostic proposer — every GT box collapsed to one class. Recall-first:
we care that objects get a box, not (here) what they are. Default model is
YOLO11-L (local weights); swap --model rtdetr-l.pt to train RT-DETR instead.
"""
import argparse
from ultralytics import YOLO, RTDETR

DATA = "/home/retrocausal-train/Documents/RPN+DinoV2/phase1/dataset/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/retrocausal-train/Documents/yolov5/data/yolo11l.pt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--name", default="yolo11l_obj")
    ap.add_argument("--imgsz", type=int, default=1024)   # high-res for the 1280x480 frames
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--rtdetr", action="store_true", help="load with the RTDETR class")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable AMP (AMP is on by default; needed to fit imgsz in 24GB)")
    ap.set_defaults(amp=True)
    args = ap.parse_args()

    Model = RTDETR if args.rtdetr else YOLO
    model = Model(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project="/home/retrocausal-train/Documents/RPN+DinoV2/phase1/runs",
        name=args.name,
        single_cls=True,
        patience=20,
        close_mosaic=10,
        seed=0,
        amp=args.amp,
        verbose=True,
    )


if __name__ == "__main__":
    main()
