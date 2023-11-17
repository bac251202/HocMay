import cv2
import numpy as np
from matplotlib import pyplot as plt


# Hàm convert_to_gray nhận một ảnh đầu vào và chuyển đổi nó thành ảnh xám
# bằng cách sử dụng hàm cv2.cvtColor từ module cv2
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Hàm compute_pca_features tính đặc trưng PCA của ảnh khuôn mặt.
# Đầu vào là ảnh khuôn mặt. Hàm chuyển đổi ảnh thành ảnh xám và sử dụng
# phân tích thành phần chính (PCA) để tính toán đặc trưng PCA của ảnh.
# Kết quả là đặc trưng PCA của ảnh khuôn mặt và được trả về.
def compute_pca_features(image):
    # Chuyển đổi ảnh màu thành ảnh xám bằng cách sử dụng hàm convert_to_gray.
    gray = convert_to_gray(image)

    # Chuyển đổi ảnh xám thành một vector 1D bằng cách sử dụng phương thức flatten
    flatten_image = gray.flatten()

    # Chuyển đổi kiểu dữ liệu của vector thành float32 bằng cách sử dụng np.float32
    flatten_image = np.float32(flatten_image)

    # Mở rộng số chiều của vector từ 1D lên thành một mảng 2D bằng cách sử dụng np.expand_dims
    flatten_image = np.expand_dims(flatten_image, axis=0)

    # Tính toán PCA trên mảng 2D sử dụng hàm cv2.PCACompute
    pca = cv2.PCACompute(flatten_image, mean=None)

    # Trích xuất các thành phần chính từ kết quả PCA và chuyển thành một vector 1D bằng cách sử dụng flatten
    pca_features = pca[0].flatten()

    # Trả về đặc trưng PCA của ảnh khuôn mặt
    return pca_features


# Hàm compare_pca_features so sánh đặc trưng PCA của hai ảnh khuôn mặt.
# Đầu vào là hai đặc trưng PCA. Hàm tính norm (độ lớn) của hiệu hai đặc trưng và trả về kết quả.
def compare_pca_features(features1, features2):
    return np.linalg.norm(features1 - features2)


# Các đường dẫn đến ảnh khuôn mặt đã biết và ảnh khuôn mặt mới cần kiểm tra
known_faces_paths = [f'train/image ({i}).JPG' for i in range(1, 39)]
unknown_image_path = "test/test (1).JPG"

# Chuẩn bị dữ liệu training
training_data = []

# Kích thước cố định cho ảnh khuôn mặt
face_size = (100, 100)

# Lặp qua các ảnh khuôn mặt đã biết
for known_face_path in known_faces_paths:
    # Đọc ảnh khuôn mặt đã biết
    known_face = cv2.imread(known_face_path)

    # Điều chỉnh kích thước của ảnh khuôn mặt đã biết
    known_face_resized = cv2.resize(known_face, face_size)

    # Trích xuất đặc trưng PCA của ảnh khuôn mặt đã biết
    known_features = compute_pca_features(known_face_resized)

    # Lưu trữ đặc trưng PCA và đường dẫn của ảnh khuôn mặt đã biết
    training_data.append((known_features, known_face_path))

# Đọc ảnh khuôn mặt mới cần kiểm tra
unknown_image = cv2.imread(unknown_image_path)

# Điều chỉnh kích thước của ảnh khuôn mặt mới
unknown_image_resized = cv2.resize(unknown_image, face_size)

# Trích xuất đặc trưng PCA của ảnh khuôn mặt mới
unknown_features = compute_pca_features(unknown_image_resized)

# Khởi tạo biến min_distance lưu trữ số thực với giá trị ban đầu là vô cùng dùng để lưu ngưỡng giá trị so sánh
min_distance = float('inf')

# Khởi tạo biến min_distance_path dùng để lưu đường dẫn ảnh kết quả
min_distance_path = ""

# So sánh đặc trưng PCA của ảnh khuôn mặt mới với các ảnh khuôn mặt đã biết
# Vòng lặp for duyệt qua bộ dữ liệu được lấy từ các ảnh đã biết,
# known_features lưu trữ giá trị đặc trưng PCA của ảnh
# known_face_path lưu trữ đường dẫn của ảnh
for known_features, known_face_path in training_data:
    # Tính toán độ lớn (distance) của sự khác biệt giữa đặc trưng PCA của ảnh
    # khuôn mặt mới (unknown_features) và ảnh khuôn mặt đã biết (known_features),
    # bằng cách sử dụng hàm compare_pca_features
    distance = compare_pca_features(unknown_features, known_features)

    # Hiển thị giá trị distance và min_distance giúp theo dõi quá trình so sánh và cập nhật giá trị min_distance
    print(distance, min_distance)

    # Kiểm tra nếu distance < min_distance thì lưu ngưỡng so sánh vào biến min_distance
    # và lưu dường dẫn của ảnh vừa lưu vào biến min_distance_path
    if distance < min_distance:
        min_distance = distance
        min_distance_path = known_face_path

# Hiển thị kết quả ảnh test và ảnh giống ảnh test nhất lên màn hình
plt.subplot(121), plt.imshow(cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)), plt.title('Ảnh Test')
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread(min_distance_path), cv2.COLOR_BGR2RGB)), plt.title(
    'Ảnh Giống Ảnh Test Nhất')
plt.show()
