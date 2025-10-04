import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Đường dẫn gốc
root_dir = Path("affectnet")
labels_csv = root_dir / "labels.csv"
train_dir = root_dir / "Train"
test_dir = root_dir / "Test"

# Tạo thư mục mới
new_train_dir = root_dir / "new_train"
new_train_dir.mkdir(exist_ok=True)

# Đọc CSV
df = pd.read_csv(labels_csv)

# Tạo subfolders cho từng label trong new_train
unique_labels = df['label'].unique()
for label in unique_labels:
    (new_train_dir / label).mkdir(exist_ok=True)

# Di chuyển/copy tất cả images từ Train và Test vào new_train dựa trên label trong CSV
# Giả sử pth là relative path như "anger/image0000006.jpg", và images ở Train/old_folder/filename hoặc Test/old_folder/filename
for _, row in df.iterrows():
    pth = row['pth']  # e.g., "anger/image0000006.jpg"
    old_folder, filename = pth.split('/', 1)
    new_label = row['label']

    # Kiểm tra trong Train trước
    old_path_train = train_dir / old_folder / filename
    if old_path_train.exists():
        src = old_path_train
    else:
        # Nếu không, kiểm tra Test
        old_path_test = test_dir / old_folder / filename
        if old_path_test.exists():
            src = old_path_test
        else:
            print(f"Warning: Image {pth} not found in Train or Test.")
            continue

    # Copy (hoặc move bằng shutil.move nếu muốn di chuyển)
    dst = new_train_dir / new_label / filename
    shutil.copy(src, dst)
    print(f"Copied {src} to {dst}")

print("All data re-distributed to new_train based on labels.csv")

# Bây giờ split 20% for Test, 80% for final Train (stratified by label)
X = df['pth'].values  # Sử dụng pth làm proxy cho samples
y = df['label'].values  # Labels

# Split stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tạo final Train và Test dirs
final_train_dir = root_dir / "final_train"
final_test_dir = root_dir / "final_test"
final_train_dir.mkdir(exist_ok=True)
final_test_dir.mkdir(exist_ok=True)

# Tạo subfolders
for label in unique_labels:
    (final_train_dir / label).mkdir(exist_ok=True)
    (final_test_dir / label).mkdir(exist_ok=True)

# Di chuyển từ new_train to final based on split
train_df = pd.DataFrame({'pth': X_train, 'label': y_train})
test_df = pd.DataFrame({'pth': X_test, 'label': y_test})

for _, row in train_df.iterrows():
    old_folder, filename = row['pth'].split('/', 1)
    new_label = row['label']
    src = new_train_dir / new_label / filename  # Vì đã ở new_train/new_label
    dst = final_train_dir / new_label / filename
    if src.exists():
        shutil.move(src, dst)  # Move để tránh duplicate

for _, row in test_df.iterrows():
    old_folder, filename = row['pth'].split('/', 1)
    new_label = row['label']
    src = new_train_dir / new_label / filename
    dst = final_test_dir / new_label / filename
    if src.exists():
        shutil.move(src, dst)

print("Data split: 80% to final_train, 20% to final_test (stratified by label)")
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
