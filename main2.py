import os
import shutil
import cv2
import dlib
import face_recognition

# 1. 사진 입력
def load_img(directory):
    images = []
    if not os.path.exists(directory):  # 입력된 디렉토리가 존재하지 않으면 경고 메시지 출력
        print(f"경로 '{directory}'에 이미지가 존재하지 않습니다.")
        return images
    
    for root, dirs, files in os.walk(directory):  # 디렉토리를 순회하며 파일 탐색
        for file in files:
            if file.endswith(('jpg', 'png', 'jpeg')):  # 이미지 파일인 경우만 처리
                images.append(os.path.join(root, file))  # 이미지 파일 경로를 리스트에 추가
    return images

# 2. 얼굴 검출 및 특징 벡터 계산
def find_target(images, target_image_paths):
    detector = dlib.get_frontal_face_detector()  # Dlib의 얼굴 검출기 초기화
    target_face_encodings = []  # 대상 얼굴의 특징 벡터를 저장할 리스트
    for target_image_path in target_image_paths:
        target_image = cv2.imread(target_image_path)  # 대상 이미지 읽기
        target_faces = detector(target_image, 1)  # 대상 이미지에서 얼굴 검출
        if len(target_faces) == 0:  # 얼굴을 찾지 못한 경우 경고 메시지 출력
            print(f"대상 이미지 '{target_image_path}'에서 얼굴을 찾을 수 없습니다.")
        else:
            # 대상 얼굴의 특징 벡터 계산하여 리스트에 추가
            target_face_encodings.extend([face_recognition.face_encodings(target_image, [(face.top(), face.right(), face.bottom(), face.left())])[0] for face in target_faces])
    return target_face_encodings

# 3. 휴지통으로 이동
def move_to_trash(images, trash_directory):
    if not os.path.exists(trash_directory):  # 휴지통 디렉토리가 존재하지 않으면 생성
        os.makedirs(trash_directory)  
    for image_path in images:
        shutil.move(image_path, trash_directory)  # 이미지를 휴지통으로 이동
        print(f"{image_path}가 휴지통으로 이동되었습니다.")

# 4. 이미지 삭제
def delete_images(trash_directory):
    while True:
        images = os.listdir(trash_directory)  # 휴지통 디렉토리에 있는 모든 파일 목록을 가져옴
        if not images:  # 휴지통이 비어있는 경우 처리
            print("휴지통에 더 이상 이미지가 없습니다.")
            break
        
        print("휴지통에 있는 이미지 목록:")
        for i, image in enumerate(images, 1):
            print(f"{i}. {image}")
        
        try:
            choice = int(input("삭제할 이미지의 번호를 선택하세요 (전체 삭제: 0, 종료: -1): "))  # 사용자 입력을 받음
            if choice == 0:  # 전체 삭제 선택
                confirm = input("휴지통의 모든 이미지를 삭제하시겠습니까? (y/n): ")
                if confirm.lower() == 'y':
                    for image in images:
                        delete_image_path = os.path.join(trash_directory, image)
                        os.remove(delete_image_path)  # 휴지통의 모든 이미지 삭제
                    print("휴지통의 모든 이미지가 삭제되었습니다.")
                else:
                    print("전체 삭제를 취소합니다.")
            elif choice == -1:  # 종료 선택
                print("프로그램을 종료합니다.")
                break
            elif choice < 1 or choice > len(images):  # 잘못된 입력 처리
                print("잘못된 번호입니다. 다시 입력하세요.")
            else:
                delete_image_path = os.path.join(trash_directory, images[choice - 1])  
                os.remove(delete_image_path)  # 선택한 이미지 삭제
                print(f"{delete_image_path}가 삭제되었습니다.")
        except ValueError:
            print("잘못된 입력입니다. 숫자를 입력하세요.")

# 메인 코드
directory = '/AI-X/201-00/dev/proj1/proimg'  # 입력 이미지 디렉토리 경로
target_image_paths = ['/AI-X/201-00/dev/proj1/vic/vic1.jpg', '/AI-X/201-00/dev/proj1/vic/vic2.jpg']  # 대상 얼굴 이미지 경로들의 리스트
trash_directory = '/AI-X/201-00/dev/proj1/trash'  # 이동시킬 휴지통 디렉토리 경로

# 과정 실행
images = load_img(directory)  # 이미지 로드
if not images:  # 이미지가 없는 경우 처리
    print("경고: 이미지가 발견되지 않았습니다.")
else:
    target_face_encodings = find_target(images, target_image_paths)  # 대상 얼굴의 특징 벡터 계산
    if not target_face_encodings:  # 대상 얼굴이 없는 경우 처리
        print("경고: 대상 얼굴과 일치하는 이미지가 발견되지 않았습니다.")
    else:
        matched_images = []  # 일치하는 이미지 경로를 저장할 리스트
        for image_path in images:  # 모든 이미지에 대해 반복
            image = cv2.imread(image_path)  # 이미지 읽기
            face_locations = face_recognition.face_locations(image)  # 얼굴 위치 찾기
            if len(face_locations) > 0:  # 얼굴이 있는 경우만 처리
                face_encodings = face_recognition.face_encodings(image, face_locations)  # 얼굴 특징 벡터 계산
                for encoding, location in zip(face_encodings, face_locations):  # 각 얼굴에 대해 반복
                    top, right, bottom, left = location  # 얼굴 위치 정보 추출
                    for target_encoding in target_face_encodings:  # 대상 얼굴과 비교
                        match = face_recognition.compare_faces([target_encoding], encoding)  # 얼굴 비교
                        if True in match:  # 일치하는 경우
                            matched_images.append(image_path)  # 일치하는 이미지 경로 저장
                            break  # 대상 얼굴과 일치하는 이미지를 찾았으므로 더 이상 비교할 필요 없음
                    if len(matched_images) > 0:  # 이미지가 하나 이상 매칭된 경우
                        break  # 이미지 분류 완료
        if not matched_images:  # 일치하는 이미지가 없는 경우
            print("경고: 대상 얼굴과 일치하는 이미지가 발견되지 않았습니다.")
        else:
            move_to_trash(matched_images, trash_directory)  # 휴지통으로 이동
            delete_images(trash_directory)  # 이미지 삭제