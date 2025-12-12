import cv2
import numpy as np
from ultralytics import YOLO


# 'yolov8n.pt' is the nano model, good for real-time performance
MODEL = YOLO('yolov8n.pt')


NAME_TO_INDEX = {v: k for k, v in MODEL.names.items()} 


# Define AR overlay data and simplified activity rules
AR_DATA = {
    'person': {
        'base_info': "Human Subject Detected",
        'activities': [
            "Sitting/Working",
            "Standing/Talking",
            "Waving/Greeting"
        ],
        'details': "Subject's inferred activity and status.",
        'color': (0, 255, 255) # Yellow
    },
    'chair': {
        'base_info': "Office Chair (AR-ID: 765)",
        'details': "Ergonomic design. Virtual link to maintenance log.",
        'color': (255, 0, 0) # Blue
    },
    'laptop': {
        'base_info': "Laptop Computer (AR-ID: 101)",
        'details': "Device Serial: ABC-123. Last logged in: 5 mins ago.",
        'color': (0, 0, 255) # Red
    },
    'book': {
        'base_info': "Detected as 'Book'",
        'details': "Overlay: Virtual Title, Author, and a 3D spinning icon.",
        'color': (0, 255, 0) # Green
    }
    # The model detects many classes; only objects in this dictionary will have custom AR data
}

# Global variable to store mouse position for interaction
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    """Updates the global mouse position on mouse move."""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def apply_ar_overlay(frame, results, mouse_pos):
    """
    Applies 2D Augmented Reality overlays and handles interaction.
    """
    global AR_DATA
    
    # Get the mouse position for interaction check
    mx, my = mouse_pos
    
    # Iterate through all detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates and convert to integers
            # The coordinate conversion fix is already correct here:
            coords = box.xyxy[0].tolist() 
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            
            # Get confidence and class name
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            cls_name = MODEL.names[cls_idx]

            # Only process objects with defined AR data and high confidence
            if cls_name in AR_DATA and conf > 0.5:
                data = AR_DATA[cls_name]
                color = data['color']
                
                # 1. Draw Bounding Box and Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Base label for all detections
                label = f"{cls_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 2. Interaction (Click/Hover Simulation) ---
                # Check if mouse is hovering over the bounding box
                is_hovered = (x1 <= mx <= x2) and (y1 <= my <= y2)

                if is_hovered:
                    # 3. Display Detailed AR Information (Virtual Overlay)
                    
                    # Define the virtual overlay area starting point
                    overlay_x = x1
                    overlay_y = y2 + 15
                    
                    # Display the AR Base Info
                    cv2.putText(frame, "--- AR DATA (INTERACTIVE) ---", (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    overlay_y += 20
                    cv2.putText(frame, data['base_info'], (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    overlay_y += 20

                    # Specific logic for "person" activity detection
                    if cls_name == 'person':
                        # Simplified Activity Detection (Rule-Based on Position/Size)
                        height = y2 - y1
                        width = x2 - x1
                        aspect_ratio = height / (width + 1e-6)
                        
                        inferred_activity = data['activities'][0] # Default: Sitting/Working
                        if aspect_ratio > 2.0: # Tall and narrow usually means standing
                            inferred_activity = data['activities'][1] # Standing/Talking
                        elif y2 < frame.shape[0] * 0.5: # Upper half of the screen
                            inferred_activity = data['activities'][2] # Waving/Greeting (Simple rule)

                        cv2.putText(frame, f"Activity: {inferred_activity}", (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        overlay_y += 20

                    # Display Detailed Information (Core Requirement 3)
                    details_lines = data['details'].split('. ')
                    for line in details_lines:
                        cv2.putText(frame, line, (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        overlay_y += 20
                        
                    # 3D Model Placeholder (Simulated)
                    if cls_name == 'book':
                        # Simulate a 3D icon using geometric shapes
                        center_x = x2 + 20
                        center_y = int((y1 + y2) / 2)
                        cv2.circle(frame, (center_x, center_y), 15, (255, 255, 0), -1) # Yellow circle
                        cv2.line(frame, (center_x - 10, center_y - 10), (center_x + 10, center_y + 10), (0, 0, 0), 2)
                        cv2.line(frame, (center_x + 10, center_y - 10), (center_x - 10, center_y + 10), (0, 0, 0), 2)
                        cv2.putText(frame, "3D V-Model", (center_x - 25, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        

    return frame

# --- Main Execution Loop ---
def main():
    # 0. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 1. Set up the window and mouse callback for interaction
    window_name = 'Real-Time AR Object Detection'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    
    
    print("--- AR App Running ---")
    print("Move your mouse over a detected object to see the AR details.")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a more intuitive 'mirror' view
        frame = cv2.flip(frame, 1)

        # 2. Run Real-Time Object Detection (YOLO)
        # We limit the classes to those we have AR data for to save processing time/clean up output
        
        # 💡 FIX 2: Use the globally defined NAME_TO_INDEX to generate integer class indices
        allowed_indices = [
            NAME_TO_INDEX[name] 
            for name in AR_DATA.keys() 
            if name in NAME_TO_INDEX
        ]
        results = MODEL(frame, classes=allowed_indices, verbose=False)

        # 3. Apply AR Overlays and Interaction Logic
        frame = apply_ar_overlay(frame, results, (mouse_x, mouse_y))

        # 4. Display the result
        cv2.imshow(window_name, frame)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()