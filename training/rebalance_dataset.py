from pathlib import Path
import shutil
import random

BASE = Path(
    r"C:\Users\sanee\Videos\websites codes\New folder\arecacare-backend\training"
)

SOURCE = BASE / "classification_data"
OUTPUT = BASE / "balanced_dataset"

CLASSES = [
    "Bud_Borer",
    "Healthy_Foot",
    "Healthy_Leaf",
    "Healthy_Nut",
    "Healthy_Trunk",
    "Mahali_Koleroga",
    "Stem_Bleeding",
    "Stem_Cracking",
    "Yellow_Leaf_Disease",
]

random.seed(42)

# Remove previous output if it exists
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

# Create folders
for split in ["train", "valid", "test"]:
    for class_name in CLASSES:
        (OUTPUT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

print("\nCreating balanced dataset split...\n")

for class_name in CLASSES:

    # Collect all images from the three existing splits
    all_images = []

    for old_split in ["train", "valid", "test"]:
        folder = SOURCE / old_split / class_name

        if folder.exists():
            for file in folder.iterdir():
                if file.suffix.lower() in [
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                ]:
                    all_images.append(file)

    # Remove duplicate filenames
    unique = {}
    for image in all_images:
        unique[image.name] = image

    all_images = list(unique.values())

    random.shuffle(all_images)

    total = len(all_images)

    if total < 3:
        print(f"WARNING: {class_name} has only {total} images")
        continue

    # Approximately 70 / 15 / 15
    test_count = max(1, round(total * 0.15))
    valid_count = max(1, round(total * 0.15))

    # Make sure at least one image remains for training
    if test_count + valid_count >= total:
        test_count = 1
        valid_count = 1

    train_count = total - valid_count - test_count

    train_images = all_images[:train_count]
    valid_images = all_images[train_count:train_count + valid_count]
    test_images = all_images[train_count + valid_count:]

    splits = {
        "train": train_images,
        "valid": valid_images,
        "test": test_images,
    }

    for split, images in splits.items():

        for index, image in enumerate(images):

            # Prevent filename conflicts
            destination = (
                OUTPUT
                / split
                / class_name
                / image.name
            )

            if destination.exists():
                destination = (
                    OUTPUT
                    / split
                    / class_name
                    / f"{index}_{image.name}"
                )

            shutil.copy2(image, destination)

    print(
        f"{class_name}: "
        f"total={total}, "
        f"train={len(train_images)}, "
        f"valid={len(valid_images)}, "
        f"test={len(test_images)}"
    )

print("\nDataset creation completed!")
print(f"\nOutput: {OUTPUT}")