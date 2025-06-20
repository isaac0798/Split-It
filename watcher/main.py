import os
from ultralytics import YOLO
import cv2

class GlassDetectorTester:
    def __init__(self, model_path='runs/detect/glass_detector/weights/best.pt', results_dir='results'):
        self.model = YOLO(model_path)
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Loaded model from: {model_path}")
        print(f"Results will be saved to: {self.results_dir}")
        
    def test_single_image(self, image_path):
        """Test detection on a single image and save to results folder"""
        results = self.model(image_path)
        
        # Get image filename without path
        image_filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_filename)[0]
        
        # Display results
        for result in results:
            # Plot bounding boxes on image
            annotated_image = result.plot()
            
            # Create output filename
            output_filename = f"{name_without_ext}_detected.jpg"
            output_path = os.path.join(self.results_dir, output_filename)
            
            # Save annotated image
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image to: {output_path}")
            
            # Print detections
            boxes = result.boxes
            if boxes is not None:
                print(f"Found {len(boxes)} detection(s):")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    class_name = self.model.names[class_id]
                    
                    print(f"  - {class_name} (confidence: {confidence:.2f})")
            else:
                print("No glasses detected")
    
if __name__ == "__main__":
    # Initialize tester
    tester = GlassDetectorTester()
    
    tester.test_single_image("./data/successful_attempt.png")