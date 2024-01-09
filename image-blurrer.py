import os
import cv2
import numpy as np
import sys
import os

# Jalur prototxt model Caffe
prototxt_path = "model-data/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "model-data/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# berikan video path sebagai argumen
image_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# memuat gambar yang akan diuji
image = cv2.imread(image_path)

# Ekstrak nama file dari image_path
filename = os.path.basename(image_path)

# Memisahkan nama file dan ekstensi
name, extension = os.path.splitext(filename)

# Gabungkan direktori output & nama file yang ditambahkan akhiran "_blurred"
output_image_path = os.path.join(
    output_directory, f"{name}_blurred{extension}")

# Dapatkan lebar dan tinggi gambar
h, w = image.shape[:2]

# ukuran kernel guassian blur tergantung pada lebar dan tinggi gambar asli
kernel_width = (w//7) | 1
kernel_height = (h//7) | 1

# memproses gambar: mengubah ukuran dan melakukan pengurangan rata-rata
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# mengatur gambar ke dalam input jaringan saraf
model.setInput(blob)

# melakukan inferensi dan mendapatkan hasilnya
output = np.squeeze(model.forward())

# looping dengan parameters
for i in range(0, output.shape[0]):
    face_accuracy = output[i, 2]
    # dapatkan tingkat akurasi wajah
    # jika akurasi wajah lebih dari 40%, maka buramkan kotak pembatas (wajah)
    if face_accuracy > 0.4:
        # dapatkan koordinat kotak sekitarnya dan tingkatkan ukuran ke gamnbar asli
        box = output[i, 3:7]*np.array([w, h, w, h])
        # ubah menjadi bilangan bulat
        start_x, start_y, end_x, end_y = box.astype(np.int64)
        # dapatkan gambar wajah
        face = image[start_y:end_y, start_x:end_x]
        # terapkan gaussian blur ke wajah ini
        face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        # masukkan wajah yang kabur ke gambar asli
        image[start_y:end_y, start_x:end_x] = face

# Setting up the width and height of the image in Windows
w = 480
h = 620

# Setting up the window size based on the original image
cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The results", w, h)

cv2.imshow("The results", image)
cv2.waitKey(0)
cv2.imwrite(output_image_path, image)
