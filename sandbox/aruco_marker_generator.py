import cv2
import numpy as np
import argparse
import sys

def generate_aruco_sheet():
    parser = argparse.ArgumentParser(description="Generate an A4 printable image with ArUco markers.")
    parser.add_argument("--size", type=float, required=True, help="Size of each marker in millimeters (mm)")
    parser.add_argument("--grid", type=int, required=True, choices=[4, 5, 6, 7], help="ArUco grid bits (e.g., 4 for 4x4, 5 for 5x5)")
    parser.add_argument("--count", type=int, required=True, help="Total number of markers to generate (starting from ID 0)")
    parser.add_argument("--output", type=str, default="aruco_a4_print.png", help="Output filename (default: aruco_a4_print.png)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the output image (default: 300)")

    args = parser.parse_args()

    # A4 dimensions in mm
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297

    # Conversion factor
    pixels_per_mm = args.dpi / 25.4
    
    # Image size in pixels
    width_px = int(A4_WIDTH_MM * pixels_per_mm)
    height_px = int(A4_HEIGHT_MM * pixels_per_mm)
    
    # Marker size in pixels
    marker_size_px = int(args.size * pixels_per_mm)
    
    # Dictionary mapping
    dictionary_map = {
        4: cv2.aruco.DICT_4X4_50,
        5: cv2.aruco.DICT_5X5_50,
        6: cv2.aruco.DICT_6X6_50,
        7: cv2.aruco.DICT_7X7_50
    }
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_map[args.grid])
    
    # Create white canvas
    canvas = np.ones((height_px, width_px), dtype=np.uint8) * 255
    
    # Margins and spacing in mm
    margin_mm = 15
    spacing_mm = 10
    
    margin_px = int(margin_mm * pixels_per_mm)
    spacing_px = int(spacing_mm * pixels_per_mm)
    
    current_x = margin_px
    current_y = margin_px
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * (args.dpi / 300)
    font_thickness = 1
    
    print(f"Generating {args.count} markers of size {args.size}mm ({marker_size_px}px) on an A4 sheet...")

    for i in range(args.count):
        # Generate marker
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size_px)
        
        # Check if it fits horizontally
        if current_x + marker_size_px > width_px - margin_px:
            current_x = margin_px
            current_y += marker_size_px + spacing_px + int(5 * pixels_per_mm) # extra space for text
            
        # Check if it fits vertically
        if current_y + marker_size_px > height_px - margin_px:
            print(f"Warning: Only {i} markers fit on a single A4 page. Stopping at index {i-1}.")
            break
            
        # Paste marker onto canvas
        canvas[current_y:current_y+marker_size_px, current_x:current_x+marker_size_px] = marker_img
        
        # Add ID label below the marker
        label = f"ID: {i}"
        label_pos = (current_x, current_y + marker_size_px + int(4 * pixels_per_mm))
        cv2.putText(canvas, label, label_pos, font, font_scale, (0,), font_thickness, cv2.LINE_AA)
        
        # Move to next position
        current_x += marker_size_px + spacing_px

    # Save output
    cv2.imwrite(args.output, canvas)
    print(f"Successfully saved printable sheet to: {args.output}")

if __name__ == "__main__":
    '''
    python3 sandbox/aruco_marker_generator.py --size 50 --grid 7 --count 6 --output /home/yuzeren/sudo/beta_setup_fact_c/sandbox/test_aruco.png
    '''
    
    if len(sys.argv) == 1:
        print("Usage example: python aruco_marker_generator.py --size 40 --grid 4 --count 10")
        print("Defaulting to interactive mode...")
        # Optional: could add input() here if desired, but argparse is better for 'from arg'
    
    generate_aruco_sheet()
