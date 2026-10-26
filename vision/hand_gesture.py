import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)


def count_fingers(landmarks):
    count = 0

    # 검지, 중지, 약지만 사용
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14)
    ]

    for tip, pip in finger_pairs:
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

start_time = time.perf_counter()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    text = "No Hand"

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        finger_count = count_fingers(landmarks)

        if finger_count == 1:
            text = "Plate 1"

        elif finger_count == 2:
            text = "Plate 2"

        elif finger_count == 3:
            text = "Plate 3"

        else:
            text = "Invalid"

        h, w, _ = frame.shape

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

    cv2.putText(
        frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()