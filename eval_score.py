import numpy as np
import os
import logging
from sklearn.metrics import jaccard_score, f1_score
import cv2

def load_npz_file(npz_path, target_size=(256, 256)):
    """加载NPZ文件并缩放分割掩码到指定大小，归一化掩码值。"""
    data = np.load(npz_path)
    masks = data['gts']
    # 将掩码图像缩放到 target_size
    masks_resized = cv2.resize(masks, target_size, interpolation=cv2.INTER_NEAREST)
    # 归一化掩码：将所有大于0的值设为1，用于二分类处理
    # masks_resized[masks_resized > 0] = 1
    return masks_resized

def calculate_dice_and_iou_per_class(pred, true, num_classes):
    """计算每个类的Dice系数和IoU"""
    dice_scores = []
    iou_scores = []
    
    for c in range(1, num_classes):  # 跳过背景类
        pred_c = (pred == c).astype(np.uint8).flatten()
        true_c = (true == c).astype(np.uint8).flatten()

        if np.sum(true_c) == 0 and np.sum(pred_c) == 0:
            dice_scores.append(1.0)
            iou_scores.append(1.0)
        else:
            dice = f1_score(true_c, pred_c, average='binary')
            iou = jaccard_score(true_c, pred_c, average='binary')
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    # 返回每个类的平均分数
    return np.mean(dice_scores), np.mean(iou_scores)

def load_filename_mapping(mapping_file_path):
    """从映射文件中加载文件名映射"""
    mapping = {}
    with open(mapping_file_path, 'r') as f:
        for line in f:
            original, renamed = line.strip().split(' -> ')
            renamed = renamed.split('.')[0] + '.png'
            mapping[renamed] = original.split('.')[0]  # 只保留文件名前缀
    return mapping

def evaluate_segmentation(result_folder, npz_folder, jpg_folder, datasets, save_folder, target_size=(256, 256)):
    os.makedirs(save_folder, exist_ok=True)
    
    # 设置日志输出
    logging.basicConfig(filename=os.path.join(save_folder, 'evaluation_log_new.txt'), 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    for dataset in datasets:
        dataset_folder = os.path.join(result_folder, dataset)
        npz_dataset_folder = os.path.join(npz_folder, dataset)
        mapping_file = os.path.join(jpg_folder, dataset, 'filename_mapping.txt')

        # 加载映射文件
        if not os.path.exists(mapping_file):
            logging.error(f"Mapping file not found for dataset: {dataset}")
            continue
        filename_mapping = load_filename_mapping(mapping_file)

        cluster_folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]

        # print(cluster_folders)
        
        for cluster_folder in cluster_folders:
            png_files = [f for f in os.listdir(cluster_folder) if f.endswith('.png')]

            results = []
            dice_scores = []
            iou_scores = []

            for png_file in png_files:
                # 找到PNG文件对应的npz前缀
                if png_file not in filename_mapping:
                    logging.error(f"PNG file {png_file} not found in mapping for dataset {dataset}")
                    continue

                npz_prefix = filename_mapping[png_file]
                npz_file = npz_prefix + '.npz'
                pred_mask_path = os.path.join(cluster_folder, png_file)

                # 读取预测的PNG文件
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

                # 读取真实的npz文件
                original_npz_path = os.path.join(npz_dataset_folder, npz_file)
                if not os.path.exists(original_npz_path):
                    logging.error(f"Original NPZ file {original_npz_path} not found")
                    continue

                true_mask_resized = load_npz_file(original_npz_path, target_size=target_size)

                # 获取类数，确保是整数
                num_classes = int(max(np.unique(pred_mask_resized).max(), np.unique(true_mask_resized).max()) + 1)

                # 计算每类的Dice和IoU
                dice, iou = calculate_dice_and_iou_per_class(pred_mask_resized, true_mask_resized, num_classes)
                dice_scores.append(dice)
                iou_scores.append(iou)

                # 记录每个样本的分割结果
                logging.info(f"Dataset: {dataset}, Cluster: {os.path.basename(cluster_folder)}, Sample: {png_file}, Dice: {dice:.4f}, IoU: {iou:.4f}")

                # 保存每个样本的结果
                results.append(f"{png_file}: Dice={dice:.4f}, IoU={iou:.4f}")

            # 保存每个cluster的结果到txt
            result_txt_path = os.path.join(save_folder, f"{os.path.basename(result_folder)}_{dataset}_{os.path.basename(cluster_folder)}_evaluation.txt")
            with open(result_txt_path, 'w') as f:
                f.write("\n".join(results))
            logging.info(f"Results saved to {result_txt_path}")

if __name__ == "__main__":
    # 定义不同的实验结果文件夹
    experiment_folders = {
        # "full_score": "/opt/data/private/MedSAM/result/sam2_vm_score",
        # "no_sort": "/opt/data/private/MedSAM/result/sam2_vm_no_sort",
        
        # "sam2_vm_score_brightness": "/opt/data/private/MedSAM/result/sam2_vm_score_brightness",
        # "sam2_vm_score_color_histogram": "/opt/data/private/MedSAM/result/sam2_vm_score_color_histogram",
        # "sam2_vm_score_constrast": "/opt/data/private/MedSAM/result/sam2_vm_score_constrast",
        # "sam2_vm_score_edge_density": "/opt/data/private/MedSAM/result/sam2_vm_score_edge_density",
        # "sam2_vm_score_shape_similarity": "/opt/data/private/MedSAM/result/sam2_vm_score_shape_similarity",
        
        "random_single_frame": "",
        "random_uniform": ""
    }

    npz_folder = ""
    jpg_folder = ""
    save_folder = ""
    
    datasets = [
        ""
    ]

    # 对每个实验结果文件夹进行评估
    for exp_name, result_folder in experiment_folders.items():
        current_save_folder = os.path.join(save_folder, exp_name)
        evaluate_segmentation(result_folder, npz_folder, jpg_folder, datasets, current_save_folder)
