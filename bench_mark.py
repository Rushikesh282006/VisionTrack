
from ultralytics import YOLO
import time

model = YOLO("yolo26n.pt")
image_path = "data/samples/bus.jpg"

# Warm-up (excluded from measurements)
model.predict(
    source=image_path,
    device="cpu",
    imgsz=640,
    verbose=False
)

# Benchmark different image sizes
for size in [640, 416, 320]:

    times = []

    for _ in range(5):

        start = time.perf_counter()

        model.predict(
            source=image_path,
            device="cpu",
            imgsz=size,
            conf=0.25,
            save=False,
            verbose=False
        )

        times.append(time.perf_counter() - start)

    average = sum(times) / len(times)

    print(
        f"Image size: {size} | "
        f"Average inference: {average:.3f} seconds"
    )
