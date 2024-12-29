!pip install opencv-python-headless


import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For Google Colab compatibility
from pytube import YouTube  # For handling YouTube video links

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Helper function to load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Object Detection using YOLO
def detect_objects_yolo(net, output_layers, frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append((boxes[i], class_ids[i]))
    return detections

# Draw detection bounding boxes
def draw_detections(frame, detections, classes, colors):
    for box, class_id in detections:
        x, y, w, h = box
        label = f"{classes[class_id]}"
        color = colors[class_id % len(colors)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main function for Smart Surveillance System
def smart_surveillance(video_source):
    # Load YOLO model
    net, classes, output_layers = load_yolo_model()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Open video source
    cap = cv2.VideoCapture(video_source)

    # Get the video frame width and height for output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

    trackers = []  # List to store all tracked objects
    tracker_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not tracker_initialized:
            # Object detection with YOLO
            detections = detect_objects_yolo(net, output_layers, frame)
            for (box, class_id) in detections:
                x, y, w, h = box
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
                corners = cv2.goodFeaturesToTrack(roi_gray, 50, 0.01, 10)
                if corners is not None:
                    corners[:, 0, 0] += x
                    corners[:, 0, 1] += y
                    trackers.append({
                        "corners": corners,
                        "old_frame": cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        "color": colors[class_id % len(colors)],
                        "box": box
                    })
            tracker_initialized = True
        else:
            for tracker in trackers:
                old_corners = tracker["corners"]
                old_frame = tracker["old_frame"]
                new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, old_corners, None, **lk_params)
                if new_corners is None or len(new_corners) < 10:
                    # If tracking fails, remove the tracker
                    trackers.remove(tracker)
                    continue

                # Update corners and draw them
                for corner in new_corners:
                    cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, tracker["color"], -1)

                # Draw bounding box around tracked object
                centroid_x = int(np.mean(new_corners[:, 0, 0]))
                centroid_y = int(np.mean(new_corners[:, 0, 1]))
                rad = int(np.linalg.norm(new_corners[:, 0, :] - [centroid_x, centroid_y], axis=1).max())
                cv2.circle(frame, (centroid_x, centroid_y), rad, tracker["color"], 2)

                # Update for next frame
                tracker["old_frame"] = new_frame_gray
                tracker["corners"] = new_corners

        # Save the processed frame to the output video
        out.write(frame)

        # Display the frame (for debugging)
        cv2_imshow(frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Exit on ESC key
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Upload video from device")
    print("2. Provide an online video URL (YouTube supported)")

    choice = input("Enter your choice (1/2): ")
    if choice == '1':
        video_path = input("Enter the path of the video file: ")
        smart_surveillance(video_path)
    elif choice == '2':
        video_url = input("Enter the YouTube video URL: ")
        yt = YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        print("Downloading video...")
        video_path = stream.download(filename='downloaded_video.mp4')
        print("Download complete. Processing video...")
        smart_surveillance(video_path)
    else:
        print("Invalid choice. Exiting.")
