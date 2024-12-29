# Smart Surveillance System  
A robust video processing application that combines **YOLO object detection** with **Lucas-Kanade optical flow-based tracking** to monitor and analyze video streams for smart surveillance.

---

## Features  
- **Object Detection**: Utilizes YOLOv4 for real-time detection of objects in video frames.  
- **Object Tracking**: Tracks detected objects using Lucas-Kanade Optical Flow, ensuring smooth monitoring across frames.  
- **Input Flexibility**: Supports video files and YouTube URLs for seamless integration.  
- **Output Video**: Saves processed video with tracked objects as `output_video.avi`.  
- **Interactive Display**: Visualizes object detection and tracking in each frame during processing.

---

## Getting Started  

### Prerequisites  
Ensure the following libraries are installed:  
- `opencv-python`
- `numpy`
- `pytube` (for handling YouTube videos)  

Install dependencies using pip:  
```bash  
pip install opencv-python numpy pytube  
```  

### Files Required  
1. **YOLO Files**:  
   - `yolov4.weights` (pre-trained weights)  
   - `yolov4.cfg` (YOLO configuration file)  
   - `coco.names` (class labels from the COCO dataset)  

---

## Usage  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/smart-surveillance.git  
   cd smart-surveillance  
   ```  

2. Run the script:  
   ```bash  
   python smart_surveillance.py  
   ```  

3. Choose input mode:  
   - **Option 1**: Upload a video from your local device.  
   - **Option 2**: Provide a YouTube video URL.  

---

## How It Works  

1. **Detection Phase**:  
   - Detects objects in video frames using YOLOv4.  
   - Applies Non-Maximum Suppression (NMS) to eliminate redundant detections.  

2. **Tracking Phase**:  
   - Tracks detected objects across frames using Lucas-Kanade Optical Flow.  
   - Updates object positions dynamically and removes failed trackers.  

3. **Visualization**:  
   - Draws bounding boxes, labels, and key points for detected and tracked objects.  

4. **Output**:  
   - Saves the processed video as `output_video.avi`.  

---

### Example Input and Output Videos

#### Input Video
[Click to view input video](https://drive.google.com/file/d/17vTtRMgAsXipdB2jxjRcHFYVhIHr5pYa/view?usp=sharing)

#### Output Video
[Click to view output video](https://drive.google.com/file/d/1fraAjRt2pFT4yWGjf5X55UhJihzfb68Y/view?usp=sharing)


## Project Structure  

```plaintext  
|-- smart_surveillance.py     # Main script  
|-- yolov4.weights            # YOLOv4 weights  
|-- yolov4.cfg                # YOLOv4 configuration file  
|-- coco.names                # COCO class labels  
```  

---

## Future Enhancements  

- Support for **real-time processing** via live camera feeds.  
- Integration of advanced object tracking algorithms (e.g., DeepSORT).  
- Transition to **YOLOv8** for improved detection speed and accuracy.  
- User-friendly GUI for better interaction.  

---

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments  

- [YOLO](https://pjreddie.com/darknet/yolo/) for the object detection framework.  
- OpenCV for computer vision tools and functions.  
- [PyTube](https://pytube.io/en/latest/) for handling YouTube video downloads.  

Feel free to contribute by submitting pull requests or issues! 🎉
