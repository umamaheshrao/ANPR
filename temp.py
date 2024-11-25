from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model("11.jpeg", show=False)

if isinstance(results, list):
    for i, img_result in enumerate(results):
        output_path = f"processed_image_{i}.jpg"
        img_result.save(output_path)
        print(f"Processed image {i+1} saved to:", output_path)

