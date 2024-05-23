import cv2
from app import RetinaFace

class FaceDetector:
    def __init__(self, model_path):
        self.model = RetinaFace(model_file=model_path)
    
    def detect_from_camera(self):
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open the camera")
            return

        while True:
            # Read each frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Cannot read data from the camera")
                break

            # Perform face detection
            det, _ = self.model.detect(frame, input_size=(640, 480))

            # Display the results
            for bbox in det:
                x1, y1, x2, y2, score = bbox
                if score > self.model.det_thresh:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    distance = self.estimate_distance(x2 - x1)
                    cv2.putText(frame, f"Distance: {distance:.2f} m", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow('Face Detection', frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def estimate_distance(self, face_width):
        # Calculate the distance based on face width
        focal_length = 480  # Updated focal length for better understanding
        real_face_width = 0.23  # real-world width of the face in meters
        distance = (focal_length * real_face_width) / face_width
        return distance

if __name__ == '__main__':
    # Load the model and start detecting faces from the camera
    face_detector = FaceDetector(model_path='model/model.onnx')
    face_detector.detect_from_camera()
