import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def load_and_examine_image_and_grayscale(image_path):
    print("=== STEP 1: LOADING IMAGE ===")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None
    
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image shape: {img.shape}")
    
    os.makedirs('./output', exist_ok=True)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./output/gray_image.jpg', img_gray)
    print("💾 Grayscale image saved to: ./output/gray_image.jpg")
    
    return img, img_gray
  
def reduce_noise(img_gray):
    """
    Remove noise (dust, scratches, camera noise) from the image
    """
    print("\n=== STEP 2: NOISE REDUCTION ===")
    print("Why? Glass has dust, fingerprints, scratches that confuse text recognition")
    
    # Create output folder
    os.makedirs('./output', exist_ok=True)

    light_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    cv2.imwrite('./output/light_blur.jpg', light_blur)
    print("✅ Light blur (3x3): Minimal noise reduction, keeps text sharp")
    
    
    return light_blur
  
def apply_thresholding(img_blurred):
  """
  Convert grayscale to black and white using different thresholding methods
  """
  print("\n=== STEP 3: THRESHOLD ===")
  print("🎯 Goal: Convert to pure black and white for text recognition")
  
  # Create output folder
  os.makedirs('./output', exist_ok=True)
  
  _, inverted_image = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  cv2.imwrite('./output/threshold_image.jpg', inverted_image)
  print("✅ Inverted Otsu: Auto-threshold + flip colors")
  
  return inverted_image

def detect_and_draw_text_boxes(img_thresh, img_original):
    print("\n=== STEP 4: TEXT DETECTION WITH BOXES ===")
    print("🔍 Finding text and drawing boxes around it...")
    
    os.makedirs('./output', exist_ok=True)
    
    # Convert grayscale to color for drawing colored boxes
    img_boxes = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    
    try:
        # Get detailed OCR data including bounding box coordinates
        ocr_data = pytesseract.image_to_data(img_thresh, output_type=pytesseract.Output.DICT)
        
        print(f"📊 OCR found {len(ocr_data['text'])} text elements")
        
        # Filter out empty/low confidence detections
        boxes_drawn = 0
        detected_words = []
        
        for i in range(len(ocr_data['text'])):
            # Get confidence score (0-100)
            confidence = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            
            # Only draw boxes for text with decent confidence
            if confidence >= 0 and len(text) > 0:
                # Get bounding box coordinates
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Choose box color based on confidence
                if confidence > 70:
                    color = (0, 255, 0)  # Green for high confidence
                    thickness = 3
                elif confidence > 50:
                    color = (0, 255, 255)  # Yellow for medium confidence
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red for low confidence
                    thickness = 1
                
                # Draw the bounding box
                cv2.rectangle(img_boxes, (x, y), (x + w, y + h), color, thickness)
                
                # Put the text and confidence above the box
                label = f"{text} ({confidence}%)"
                cv2.putText(img_boxes, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"   📦 Box {boxes_drawn + 1}: '{text}' (confidence: {confidence}%)")
                detected_words.append((text, confidence))
                boxes_drawn += 1
        
        print(f"\n✅ Drew {boxes_drawn} bounding boxes")
        
        # Save the result
        cv2.imwrite('./output/step4_text_boxes.jpg', img_boxes)
        print("💾 Saved image with boxes: ./output/step4_text_boxes.jpg")
        
        return detected_words
        
    except Exception as e:
        print(f"❌ Error during text detection: {str(e)}")
        return None

def main():
    image_path = "./data/full_glass.jpg"
    
    result = load_and_examine_image_and_grayscale(image_path)
    
    if result is not None:
        img_bgr, img_gray = result
        print("\n🎉 Step 1 complete! We successfully loaded and displayed the image.")
        print("\nNext steps will be:")
        img_blur = reduce_noise(img_gray)
        img_threshold = apply_thresholding(img_blur)
        detect_and_draw_text_boxes(img_threshold, img_gray)
        print("- Extract text")
    else:
        print("\n❌ Please check your image path and try again.")

if __name__ == "__main__":
    main()