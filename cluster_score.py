import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil
import logging

# 设置日志
logging.basicConfig(filename='cluster_score.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 亮度、对比度、边缘密度、颜色直方图相似性、形状相似性等综合打分函数
def calculate_brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness / 255.0

def calculate_contrast_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast / 255.0

def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)

def calculate_color_histogram_similarity(ref_image, image):
    ref_hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_ref = cv2.calcHist([ref_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_ref = cv2.normalize(hist_ref, hist_ref).flatten()
    hist = cv2.normalize(hist, hist).flatten()
    score = cv2.compareHist(hist_ref, hist, cv2.HISTCMP_CORREL)
    return score

def calculate_shape_similarity(ref_image, image):
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref_moments = cv2.moments(ref_gray)
    moments = cv2.moments(gray)
    hu_moments_ref = cv2.HuMoments(ref_moments).flatten()
    hu_moments = cv2.HuMoments(moments).flatten()
    # 避免log(0)的情况
    epsilon = 1e-10
    score = -np.log(np.sum(np.abs(hu_moments_ref - hu_moments)) + epsilon)
    return score

def compute_composite_score(ref_image, image):
    brightness_weight = 0.1
    contrast_weight = 0.1
    edge_density_weight = 0.1
    color_histogram_weight = 0.3
    shape_similarity_weight = 0.4

    brightness_score = calculate_brightness_score(image)
    contrast_score = calculate_contrast_score(image)
    edge_density_score = calculate_edge_density(image)
    color_histogram_score = calculate_color_histogram_similarity(ref_image, image)
    shape_similarity_score = calculate_shape_similarity(ref_image, image)

    composite_score = (
        brightness_weight * brightness_score +
        contrast_weight * contrast_score +
        edge_density_weight * edge_density_score +
        color_histogram_weight * color_histogram_score +
        shape_similarity_weight * shape_similarity_score
    )

    return composite_score

def resize_frame(frame, target_size):
    """调整帧的大小"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def select_representative_frames_by_kmeans(frames, ref_frame_idx, num_clusters=5):
    """选择代表性帧，基于KMeans聚类，并且基于参考帧进行计算"""
    ref_frame = frames[ref_frame_idx]
    target_size = (ref_frame.shape[1], ref_frame.shape[0])  # 宽, 高
    resized_frames = [resize_frame(frame, target_size) for frame in frames]
    
    # 计算参考帧与所有其他帧的综合打分值
    composite_scores = []
    for i, frame in enumerate(resized_frames):
        if i == ref_frame_idx:  # 跳过参考帧与自身的比较
            continue
        score = compute_composite_score(ref_frame, frame)
        composite_scores.append(score)
        logging.info(f"Composite score between reference frame and frame {i+1}: {score:.4f}")
    
    composite_scores = np.array(composite_scores).reshape(-1, 1)
    
    # 使用KMeans聚类
    logging.info("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(composite_scores)
    
    # 获取每个簇的中心并找到最接近中心的帧
    selected_indices = []
    for center in kmeans.cluster_centers_:
        closest_idx = np.argmin(np.abs(composite_scores - center))
        if closest_idx not in selected_indices:  # 防止重复选择同一帧
            selected_indices.append(closest_idx)
            logging.info(f"Selected frame {closest_idx+1} as a cluster center.")
    
    # 确保聚类中心不包括仅有参考帧的聚类簇
    if ref_frame_idx in selected_indices:
        selected_indices.remove(ref_frame_idx)
    
    return sorted(set(selected_indices)), kmeans.labels_

def sort_frames_within_clusters(frames, ref_frame, labels, num_clusters):
    """对聚类中的帧按照综合打分值进行排序"""
    target_size = (ref_frame.shape[1], ref_frame.shape[0])  # 确保与参考帧相同的大小
    resized_frames = [resize_frame(frame, target_size) for frame in frames]
    
    sorted_clusters = []
    for cluster in range(num_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_frames = [resized_frames[idx] for idx in cluster_indices]
        
        # 基于参考帧对聚类中的帧进行排序
        composite_scores = [compute_composite_score(ref_frame, frame) for frame in cluster_frames]
        sorted_cluster_indices = [x for _, x in sorted(zip(composite_scores, cluster_indices), reverse=True)]
        sorted_clusters.append(sorted_cluster_indices)
        logging.info(f"Sorted frames in cluster {cluster}: {[i+1 for i in sorted_cluster_indices]}")
    
    return sorted_clusters

def save_cluster_sorting_info(sorted_clusters, dataset_name, output_folder):
    """保存聚类后排序的帧索引信息"""
    for cluster_idx, cluster_frames in enumerate(sorted_clusters):
        output_file = os.path.join(output_folder, f"{dataset_name}_cluster_{cluster_idx}_sorted.txt")
        with open(output_file, 'w') as f:
            for frame_idx in cluster_frames:
                f.write(f"{dataset_name} Frame {frame_idx+1:06d}\n")
        logging.info(f"Saved sorted frames for cluster {cluster_idx} in {output_file}")

def copy_frames_to_cluster_folders(frames, sorted_clusters, dataset_folder, output_folder):
    """将排序后的聚类结果复制到对应的文件夹中"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for cluster_idx, cluster_frames in enumerate(sorted_clusters):
        cluster_folder = os.path.join(output_folder, f"cluster_{cluster_idx}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        for frame_idx in cluster_frames:
            src_image_path = os.path.join(dataset_folder, f"{frame_idx+1:06d}.jpg")
            dst_image_path = os.path.join(cluster_folder, f"{frame_idx+1:06d}.jpg")
            shutil.copy(src_image_path, dst_image_path)
        
        # 标记聚类中心
        cluster_center = cluster_frames[0]
        with open(os.path.join(cluster_folder, "cluster_center.txt"), "w") as f:
            f.write(f"Cluster center frame: {cluster_center+1:06d}.jpg")
        logging.info(f"Cluster {cluster_idx} center is frame {cluster_center+1}.")

def process_datasets_for_clustering(npz_dataset_folders, jpg_output_folder, clustered_output_folder, ref_frame_idx=0, num_clusters=5):
    """对所有数据集进行聚类处理"""
    for dataset_folder in npz_dataset_folders:
        dataset_name = os.path.basename(dataset_folder)
        image_folder = os.path.join(jpg_output_folder, dataset_name)
        output_folder = os.path.join(clustered_output_folder, dataset_name)
        
        # 读取图像序列并过滤掉无法加载的图像
        num_images = len(os.listdir(image_folder))
        frames = []
        for i in range(num_images):
            image_path = os.path.join(image_folder, f"{i+1:06d}.jpg")
            frame = cv2.imread(image_path)
            if frame is not None:
                frames.append(frame)
            else:
                logging.warning(f"Failed to load image: {image_path}")

        if not frames:
            logging.warning(f"No valid images found in {image_folder}. Skipping dataset.")
            continue

        # 选择代表性帧
        selected_indices, labels = select_representative_frames_by_kmeans(frames, ref_frame_idx, num_clusters)
        
        # 对每个簇中的帧进行排序
        ref_frame = frames[ref_frame_idx]
        sorted_clusters = sort_frames_within_clusters(frames, ref_frame, labels, num_clusters)
        
        # 将帧复制到对应的文件夹中
        copy_frames_to_cluster_folders(frames, sorted_clusters, image_folder, output_folder)

        # 保存聚类后的排序信息
        save_cluster_sorting_info(sorted_clusters, dataset_name, output_folder)


# Example usage
npz_dataset_folders = [
""
]
jpg_output_folder = "/data/val_jpg"
clustered_output_folder = "clustered_output_score"

process_datasets_for_clustering(npz_dataset_folders, jpg_output_folder, clustered_output_folder, ref_frame_idx=0, num_clusters=5)
