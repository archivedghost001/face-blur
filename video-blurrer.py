import cv2
import numpy as np
import time
import sys
import os

# Jalur prototxt model Caffe
prototxt_path = "model-data/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "model-data/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# berikan video path sebagai argumen
video_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# capture frames from video
capture = cv2.VideoCapture(video_path)

# ekstrak nama file dari video_path
filename = os.path.basename(video_path)

# memisahkan nama file dan ekstensi
the_name, extension = os.path.splitext(filename)

# membuat four-character kode yang dipakai untuk mengetahui video codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video, image = capture.read()
print(image.shape)
output_video_path = cv2.VideoWriter(
    os.path.join(output_directory, f"{the_name}_blurred{extension}"),
    fourcc,
    20.0,
    (image.shape[1], image.shape[0])
)

# buat forever looping
while True:
    start = time.time()
    captured, image = capture.read()

    # get width and height of the image
    if not captured:
        break
    h, w = image.shape[:2]
    # ukuran kernel guassian blur tergantung pada lebar dan tinggi gambar asli
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1

    # memproses gambar: mengubah ukuran dan melakukan pengurangan rata-rata
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # mengatur gambar ke dalam input jaringan saraf
    model.setInput(blob)

    # melakukan inferensi dan mendapatkan hasilnya
    output = np.squeeze(model.forward())

    # looping dengan parameters
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # dapatkan tingkat akurasi wajah
        # jika akurasi wajah lebih dari 40%, maka buramkan kotak pembatas (wajah)
        if confidence > 0.4:
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
    # mengatur lebar & tinggi
    width = 480
    height = 640

    # mengatur ukuran Window sesua dengan gambar asli
    cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("The results", width, height)

    # menampilkan hasilnya
    cv2.imshow("The results", image)
    # tombol close jika tekan "q"
    if cv2.waitKey(1) == ord("q"):
        break
    time_elapsed = time.time() - start
    fps = 1 / time_elapsed
    print("FPS", fps)
    output_video_path.write(image)

cv2.destroyAllWindows()
capture.release()
output_video_path.release()
