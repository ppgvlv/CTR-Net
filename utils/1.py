import os
import shutil

# === 输入路径配置 ===
train_txt = 'E:/数据集/IRSTD-1k-6-4/IRSTD-1k/trainval.txt'
test_txt = 'E:/数据集/IRSTD-1k-6-4/IRSTD-1k/test.txt'

image_dir = 'E:/数据集/IRSTD-1k-6-4/IRSTD-1k/images'     # 原图目录
mask_dir = 'E:/数据集/IRSTD-1k-6-4/IRSTD-1k/masks'       # 掩码目录

output_dir = 'E:/数据集/IRSTD-1k-6-4/IRSTD-1k'   # 输出根目录

# === 读取编号列表函数 ===
def load_ids(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# === 拷贝文件函数 ===
def copy_files(ids, subset):
    for id_ in ids:
        img_src = os.path.join(image_dir, id_ + '.png')
        mask_src = os.path.join(mask_dir, id_ + '.png')

        img_dst = os.path.join(output_dir, subset, 'images', id_ + '.png')
        mask_dst = os.path.join(output_dir, subset, 'masks', id_ + '.png')

        # 创建输出目录
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(mask_dst), exist_ok=True)

        # 拷贝
        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"[警告] 文件缺失：{id_}")

# === 主流程 ===
train_ids = load_ids(train_txt)
test_ids = load_ids(test_txt)

copy_files(train_ids, 'train')
copy_files(test_ids, 'test')

print("✅ 数据划分完成！")
