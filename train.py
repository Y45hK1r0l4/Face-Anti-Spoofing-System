from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')  // yolo v8 neno

def main():
    // training the data 
    model.train(data='Dataset/SplitData/data.yml', epochs=3)

if __name__ == '__main__':
    main()
