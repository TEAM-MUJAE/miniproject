from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import face_recognition

app = FastAPI()

# 이미지와 처리된 파일을 저장하는 디렉토리
ximg_dir = "ximg"
images_dir = "images"
trash_dir = "trash"


# 디렉토리 이미지 필터링
def filter_images_with_faces(directory):
    filtered_images = []
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                filtered_images.append(file_path)
    except Exception as e:
        return JSONResponse(content={"error": f"Error filtering images: {str(e)}"}, status_code=500)

    return filtered_images

"""
    참조 이미지와 유사한 얼굴을 가진 이미지를 주어진 리스트에서 찾습니다.

    Args:
        x_img_path (str): 참조 이미지의 파일 경로.
        images_list (list): 비교할 이미지 파일 경로의 리스트.

    Returns:
        list: 유사한 얼굴을 가진 파일 경로의 리스트.
"""

def find_similar_images(x_img_path, images_list):
    x_image = face_recognition.load_image_file(x_img_path)
    try:
        x_face_encoding = face_recognition.face_encodings(x_image)[0]
    except IndexError:
        return JSONResponse(content={"error": "No face found in the uploaded image."}, status_code=400)

    similar_images = []

    for image_path in images_list:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:
            if face_recognition.compare_faces([x_face_encoding], face_encoding[0], tolerance=0.4)[0]:
                similar_images.append(image_path)

    return similar_images

@app.post("/process_image/")
async def process_image(x_file: UploadFile = File(...)):

    # 업로드된 x.jpg 파일을 ximg 디렉터리에 저장
    x_img_path = os.path.join(ximg_dir, "x.jpg")
    try:
        with open(x_img_path, "wb") as x_img:
            x_img.write(x_file.file.read())
    except Exception as e:
        return JSONResponse(content={"error": f"Error saving uploaded image: {str(e)}"}, status_code=500)

    # 이미지 디렉터리에서 얼굴이 있는 이미지를 필터링
    filtered_images = filter_images_with_faces(images_dir)
    if isinstance(filtered_images, JSONResponse):
        return filtered_images

    # 유사한 이미지를 찾아 휴지통으로 이동
    similar_images = find_similar_images(x_img_path, filtered_images)
    for img_path in similar_images:
        trash_img_path = os.path.join(trash_dir, os.path.basename(img_path))
        try:
            os.rename(img_path, trash_img_path)
        except Exception as e:
            return JSONResponse(content={"error": f"Error moving image to trash: {str(e)}"}, status_code=500)

    return JSONResponse(content={"message": "Image processing complete."})

