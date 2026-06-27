import cv2

def analizeVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    flag = True
    flag2 = 0
    while True:
        success, frame = cap.read()
        if not success: break
        if flag2 > 0:
            flag2 -= 1
            count += 1
            continue
        if flag and flag2 == 0:
            cv2.putText(frame, f"frame: {count:03d}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(f"frame",frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == 32:
                flag2 = 9
            if key == 13:
                flag = False
            
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total: {count}")

#string = str(input("Nome:"))

#analizeVideo(f"Data/dataset/{string}.avi")

analizeVideo(f"Data/dataset/chute07cam1.avi")