from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='Dataset/SplitData/data.yml', epochs=3)

if __name__ == '__main__':
    main()