python train.py --workers=4 --epochs=15 --batch-size=32 --learning-rate=0.0001 --weight-decay=0 --pretrained=True
python predict.py --model=weights/best.pt --workers=4 --batch-size=32