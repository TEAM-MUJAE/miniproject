# 종합본
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import face_recognition
import uvicorn
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 디렉토리 설정
ximg_dir = "ximg" #기준 이미지 폴더
images_dir = "static/images" # 분류하고자하는 이미지 폴더
trash_dir = "trash" #휴지통 폴더


# 필요한 디렉토리 생성
os.makedirs(ximg_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(trash_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main():
    # 메인 페이지 렌더링
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/upload_image")
async def upload_image(x_file: UploadFile = File(...)):
    # 클라이언트가 보낸 이미지 저장
    x_img_path = os.path.join(ximg_dir, x_file.filename)
    with open(x_img_path, "wb") as buffer:
        shutil.copyfileobj(x_file.file, buffer)
        
    x_image = face_recognition.load_image_file(x_img_path)
    x_face_encoding = face_recognition.face_encodings(x_image)
    if len(x_face_encoding) == 0:
        return JSONResponse(content={"error": "No human face found in the image. Please upload an image with a human face."}, status_code=400)

    # 유사한 이미지 찾기
    similar_images = find_similar_images(x_img_path)
    
    if not similar_images:
        # 매칭되는 이미지가 없을 경우 사용자에게 알림 메시지 보내기
        return JSONResponse(content={"error": "매칭되는 이미지가 없습니다."}, status_code=404)

    a_dir = os.path.join(images_dir, os.path.splitext(x_file.filename)[0])  # 'a' 폴더 경로
    os.makedirs(a_dir, exist_ok=True)  # 'a' 폴더 생성
    move_images(similar_images, a_dir)  # 유사한 이미지를 'a' 폴더로 이동

    return JSONResponse(content={"message": f"Images similar to {x_file.filename} have been classified and moved to {a_dir}."})

def move_images(image_paths, destination):
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        dest_img_path = os.path.join(destination, basename)
        shutil.move(img_path, dest_img_path)

def find_similar_images(x_img_path):
    x_image = face_recognition.load_image_file(x_img_path)
    try:
        x_face_encoding = face_recognition.face_encodings(x_image)[0]
    except IndexError:
        raise HTTPException(status_code=400, detail="No face found in the uploaded image.")
    
    similar_images = []
    # images_dir 내의 모든 파일에 대해 순회
    for root, dirs, files in os.walk(images_dir):
        for filename in files:
            # 이미지 파일 확장자 확인
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces([x_face_encoding], face_encoding)
                        if True in matches:
                            similar_images.append(image_path)
                            break  # 첫 번째 일치하는 이미지에서 멈춤
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")
    
    return similar_images


# 이미지 휴지통으로 이동
def move_images_to_trash(image_paths):
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        trash_img_path = os.path.join(trash_dir, basename)
        shutil.move(img_path, trash_img_path)


@app.post("/move_to_trash")
async def move_to_trash(x_file: UploadFile = File(...)):
    # 클라이언트가 보낸 이미지 저장
    x_img_path = os.path.join(ximg_dir, x_file.filename)
    with open(x_img_path, "wb") as buffer:
        shutil.copyfileobj(x_file.file, buffer)
        
    x_image = face_recognition.load_image_file(x_img_path)
    x_face_encoding = face_recognition.face_encodings(x_image)
    if len(x_face_encoding) == 0:
        return HTTPException(status_code=400, detail="No face found in the uploaded image.")

    # 유사한 이미지 찾기
    similar_images = find_similar_images(x_img_path)
    
    if not similar_images:
        # 매칭되는 이미지가 없을 경우 사용자에게 알림 메시지 보내기
        return HTTPException(status_code=404, detail="No face found in the uploaded image.")


    # 유사한 이미지를 휴지통으로 이동
    move_images_to_trash(similar_images)

    return JSONResponse(content={"message": f"Images similar to {x_file.filename} have been moved to the trash."})

# 휴지통을 비우는 함수
@app.post("/empty_trash")
def empty_trash():
    # 휴지통에 있는 모든 이미지 삭제
    trash_items = os.listdir(trash_dir)
    if not trash_items: # 휴지통이 비어있는 경우
        raise HTTPException(status_code=404, detail="휴지통이 이미 비어 있습니다.")

    for filename in os.listdir(trash_dir):
        file_path = os.path.join(trash_dir, filename)
        os.remove(file_path)
    return JSONResponse(content={"message": "휴지통이 성공적으로 비워졌습니다."})


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000,reload=True)