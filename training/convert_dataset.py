import os
import shutil
from pathlib import Path

# Dataset location
DATASET = Path(
    r"C:\Users\sanee\Videos\websites codes\New folder\arecacare-backend\training\dataset"
)

# Output classification dataset
OUTPUT = Path(
    r"C:\Users\sanee\Videos\websites codes\New folder\arecacare-backend\training\classification_data"
)

CLASS_NAMES = [
    "Bud_Borer",
    "Healthy_Foot",
    "Healthy_Leaf",
    "Healthy_Nut",
    "Healthy_Trunk",
    "Mahali_Koleroga",
    "Stem_Bleeding",
    "Stem_Cracking",
    "Yellow_Leaf_Disease"
]

# Create output folders
for split in ["train", "valid", "test"]:
    for class_name in CLASS_NAMES:
        (OUTPUT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


def convert_split(split):
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    if not images_dir.exists():
        print(f"ERROR: Missing {images_dir}")
        return

    if not labels_dir.exists():
        print(f"ERROR: Missing {labels_dir}")
        return

    count = 0

    for image_file in images_dir.iterdir():

        if image_file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        label_file = labels_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        # Copy image into every class represented
        # in its YOLO annotation.
        classes_found = set()

        for line in lines:
            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])

            if 0 <= class_id < len(CLASS_NAMES):
                classes_found.add(class_id)

        for class_id in classes_found:

            class_name = CLASS_NAMES[class_id]

            destination = (
                OUTPUT
                / split
                / class_name
                / image_file.name
            )

            # Avoid overwriting files with same name
            if destination.exists():
                destination = (
                    OUTPUT
                    / split
                    / class_name
                    / f"{image_file.stem}_{class_id}{image_file.suffix}"
                )

            shutil.copy2(image_file, destination)
            count += 1

    print(f"{split}: {count} images copied")


print("\nStarting dataset conversion...\n")

convert_split("train")
convert_split("valid")
convert_split("test")

print("\nConversion completed!")
print(f"\nOutput location:\n{OUTPUT}")

print("\nClass folders:")
for class_name in CLASS_NAMES:
    print(" -", class_name)