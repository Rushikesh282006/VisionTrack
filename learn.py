def kitti_bbox_to_yolo(
    bbox: tuple[float, float, float, float], 
    img_width: int, 
    img_height: int
) -> tuple[float, float, float, float]:
    """
    Converts KITTI pixel bounding box to normalized YOLO format.
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.
        img_width: Physical width of the image in pixels.
        img_height: Physical height of the image in pixels.
        
    Returns:
        (x_center, y_center, width, height) normalized to [0.0, 1.0].
    """
    x_min,y_min,x_max,y_max = bbox

    if x_min >= x_max or y_min >= y_max :
        return None

    width = x_max - x_min
    height = y_max - y_min

    x_center =  x_min + width/2
    y_center =  y_min + height/2

    x_c_norm = x_center / img_width
    y_c_norm = y_center / img_height

    width = width / img_width
    height = height / img_height
    
    return (x_c_norm,y_c_norm,width,height) 


def parse_kitti_line(
    line: str, 
    class_mapping: dict[str, int]
) -> tuple[int, tuple[float, float, float, float]] | None:
    
    line_list = line.split()

    if len(line_list) < 15:
        return None    
    elif line_list[0] not in class_mapping:
        return None
    else:    
        list_key = class_mapping[line_list[0]]
        return (list_key,(float(line_list[4]),float(line_list[5]),float(line_list[6]),float(line_list[7]),))
    

def convert_kitti_file(
    file_path: str,
    img_width: int,
    img_height: int,
    class_mapping: dict[str, int]
) -> list[str]:
    
        yolo_lines = []
        
        with open(file_path, "r") as file:
            for line in file:
                temp = parse_kitti_line(line,class_mapping)
                if temp != None:
                    yolo_lines.append(temp)

        file.close()

        yolo_str = []

        for line in yolo_lines:
            if line is not None:
                temp = kitti_bbox_to_yolo(line[1],img_width,img_height)
                yolo_str.append(f"{line[0]} {temp[0]} {temp[1]} {temp[2]} {temp[3]}")

        return yolo_str
    

import random

def split_dataset(
    sample_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict[str, list[str]]:
    """
    Splits sample IDs into train, val, and test sets reproducibly.
    
    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    if train_ratio+test_ratio+val_ratio != 1 :
        raise ValueError("The Sum of ratios should be 1")

    random.seed(seed)

    shuffled_data = sample_ids.copy()
    random.shuffle(shuffled_data)
    n  = len(shuffled_data)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train : n_train + n_val]
    test_data = shuffled_data[n_train + n_val :]
    data_dict = {"train_data":train_data,"test_data":test_data,"val_data":val_data}

    return data_dict

from pathlib import Path

# if __name__ == "__main__":
#     labels_dir = Path("data/raw/kitti/training/label_2")
    
#     # Grab all 7,481 file stems (e.g. '000000', '000001', ...)
#     sample_ids = [p.stem for p in sorted(labels_dir.glob("*.txt"))]
    
#     splits = split_dataset(sample_ids)
    
#     print("\n--- KITTI DATASET PARTITIONING ---")
#     print(f"Total files: {len(sample_ids)}")
#     print(f"Train split: {len(splits['train_data'])} ({len(splits['train_data'])/len(sample_ids)*100:.1f}%)")
#     print(f"Val split:   {len(splits['val_data'])} ({len(splits['val_data'])/len(sample_ids)*100:.1f}%)")
#     print(f"Test split:  {len(splits['test_data'])} ({len(splits['test_data'])/len(sample_ids)*100:.1f}%)")






def yolo_to_pixel_bbox(
    yolo_box: tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> tuple[int, int, int, int]:
    """
    Converts normalized YOLO coordinates back into pixel coordinates (xmin, ymin, xmax, ymax).
    
    Args:
        yolo_box: (x_center, y_center, width, height) normalized in [0.0, 1.0].
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        
    Returns:
        (xmin, ymin, xmax, ymax) as integer pixel coordinates.
    """
    
    x_c_px = yolo_box[0] * img_width

    y_c_px = yolo_box[1] * img_height

    w_px = yolo_box[2] * img_width

    h_px = yolo_box[3] * img_height

    x_min = x_c_px - w_px/2

    y_min = y_c_px - h_px/2

    x_max = x_min + w_px

    y_max = y_min + h_px

    return (int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max)))    


import cv2
import numpy as np

def render_yolo_boxes(
    image: np.ndarray,
    yolo_boxes: list[tuple[int, tuple[float, float, float, float]]],
    class_names: dict[int, str]
) -> np.ndarray:
    """
    Renders bounding boxes and class labels onto an image.
    
    Args:
        image: BGR numpy image array.
        yolo_boxes: List of (class_id, (x_center, y_center, width, height)).
        class_names: Dict mapping class_id -> class_name (e.g. {0: "car", 1: "truck"}).
        
    Returns:
        Annotated BGR image array.
    """
    img_h,img_w = image.shape[:2]

    for class_id,bbox_norm in yolo_boxes:
        xmin,ymin,xmax,ymax = yolo_to_pixel_bbox(bbox_norm,img_w,img_h)
        cv2.rectangle(
            image, 
            pt1=(xmin, ymin), 
            pt2=(xmax, ymax), 
            color=(0, 255, 0), 
            thickness=2
        )

        cls_name = class_names.get(class_id, f"class_{class_id}")
        label_text = f"{cls_name} ({class_id})"

        cv2.putText(
            image,
            text=label_text,
            org=(xmin, max(ymin - 10, 15)), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA             
        )
    
    return image


if __name__ == "__main__":
    from pathlib import Path

    # 1. Image dimensions
    W, H = 1242, 375

    # 2. Class mapping
    names = {0: "car", 1: "truck", 2: "bus", 3: "pedestrian", 4: "cyclist"}
    
    # 3. Create a blank dark gray canvas (simulating a road frame)
    canvas = np.full((H, W, 3), 40, dtype=np.uint8)

    # 4. Our 3 converted YOLO boxes from 000001.txt: (class_id, (xc, yc, w, h))
    test_boxes = [
        (1, (0.494831, 0.460867, 0.024428, 0.087600)),  # Truck
        (0, (0.326667, 0.512880, 0.029130, 0.057547)),  # Car
        (4, (0.549750, 0.477173, 0.009968, 0.079947)),  # Cyclist
    ]

    # 5. Render
    annotated = render_yolo_boxes(canvas, test_boxes, names)

    # 6. Save image to outputs
    output_path = Path("outputs/test_render.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved visual verification image to: {output_path}")


def validate_yolo_line(
    line: str, 
    valid_class_ids: set[int] = {0, 1, 2, 3, 4}
) -> tuple[bool, str]:
    """
    Validates a single YOLO annotation line.
    
    Returns:
        (True, "Valid") if the line passes all integrity checks.
        (False, "Error reason") if any check fails.

    """

    line_list = line.split()
    
    try:
        class_id = int(line_list[0])
        xc, yc, w, h = float(line_list[1]), float(line_list[2]), float(line_list[3]), float(line_list[4])
    except ValueError:
        return False, "Failed to parse numbers"

    # 1. Class ID check
    if class_id not in valid_class_ids:
        return False, f"Class ID {class_id} is invalid"

    # 2. Positive dimensions check (width & height must be > 0)
    if w <= 0.0 or h <= 0.0:
        return False, "Box dimensions must be strictly positive"

    # 3. Normalized range check [0.0, 1.0]
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and w <= 1.0 and h <= 1.0):
        return False, "Coordinates or dimensions exceed [0.0, 1.0]"

    return True, "Valid"




