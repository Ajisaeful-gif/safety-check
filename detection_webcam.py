from ultralytics import YOLO
import cv2
import math

def detect_objects(path_x):
    video_capture = path_x
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    model=YOLO("YOLO-Weights/best_model_deteksi.pt")
    classNames = ['Glasses','Gloves','Helmet','Mask','Safety Vest']
    
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        
        # Lists to store the presence of each type of PPE
        glasses_detected = False
        gloves_detected = False
        mask_detected = False

        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                if class_name == 'Glasses' and conf > 0.5:
                    glasses_detected = True
                    color = (222, 82, 175)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1,lineType=cv2.LINE_AA)

                elif class_name == 'Gloves' and conf > 0.8:
                    gloves_detected = True
                    color = (255, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                elif class_name == 'Mask' and conf > 0.85:
                    mask_detected = True
                    color = (0, 204, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Check if all three types of PPE are detected
        if glasses_detected and gloves_detected and mask_detected:
            apd_status = "APD Sesuai"
            color = (0, 255, 0)  # Green
        else:
            apd_status = "APD Tidak Sesuai"
            color = (0, 0, 255)  # Red

        # Display the status on the image with the specified color
        cv2.putText(img, apd_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        yield img

    cv2.destroyAllWindows()
