import cv2
import os

def match_images(small_image_folder, large_image_folder):
    # 获取小图像文件夹中所有图像的文件列表
    small_images = [os.path.join(small_image_folder, file) for file in os.listdir(small_image_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # 使用ORB特征检测器和描述器
    orb = cv2.ORB_create()

    for small_image_path in small_images:
        # 读取小图像
        small_image = cv2.imread(small_image_path, 0)

        # 初始化最佳匹配变量
        best_match = None
        best_match_score = float('inf')

        # 遍历大图像文件夹中的图像
        for large_image_path in os.listdir(large_image_folder):
            # 读取大图像
            large_image = cv2.imread(os.path.join(large_image_folder, large_image_path), 0)

            # 寻找关键点和描述符
            kp1, des1 = orb.detectAndCompute(small_image, None)
            kp2, des2 = orb.detectAndCompute(large_image, None)

            # 使用暴力匹配器
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # 计算匹配得分
            match_score = sum([match.distance for match in matches])

            # 更新最佳匹配
            if match_score < best_match_score:
                best_match_score = match_score
                best_match = large_image_path

        # 打印匹配结果
        print(f"Best match for {os.path.basename(small_image_path)}: {best_match} (Score: {best_match_score})")

if __name__ == "__main__":
    # 替换为小图像和大图像的文件夹路径
    small_image_folder = "E:\Desktop\Study\BU\Sem1\AI\project\/2"
    large_image_folder = "E:\Desktop\Study\BU\Sem1\AI\project\/1"

    match_images(small_image_folder, large_image_folder)
