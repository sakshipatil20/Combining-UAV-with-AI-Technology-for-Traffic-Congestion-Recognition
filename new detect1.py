import argparse
import sys
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from gpiozero import LED
from time import sleep

# Initialize LEDs
red_led = LED(14)  # Replace 17 with the actual GPIO pin for the red LED
yellow_led = LED(15)  # Replace 27 with the actual GPIO pin for the yellow LED
green_led = LED(18)  # Replace 22 with the actual GPIO pin for the green LED

def process_image(image, detector):
    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on the input image
    image = utils.visualize(image, detection_result)

    # Count and print cars
    car_count = utils.count_and_print_cars(detection_result)
    print("Total Car count:", car_count)

    # Traffic system logic
    if car_count >= 7:
        print("High Traffic")
        red_led.on()
        yellow_led.off()
        green_led.off()
     #   sleep(60)  # High traffic time: 1 minute
        red_led.off()
        yellow_led.off()
        green_led.off()
    elif 4 <= car_count <= 6:
        print("Medium Traffic")
        red_led.off()
        yellow_led.on()
        green_led.off()
      #  sleep(30)  # Medium traffic time: 30 seconds
        red_led.off()
        yellow_led.off()
        green_led.off()
    elif 1 <= car_count <= 3:
        print("Low Traffic")
        red_led.off()
        yellow_led.off()
        green_led.on()
       # sleep(10)  # Low traffic time: 10 seconds
        red_led.off()
        yellow_led.off()
        green_led.off()
    else:
        print("No Traffic")
        red_led.off()
        yellow_led.off()
        green_led.off()

    return image

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    counter, fps = 0, 0
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=30, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
        cv2.imshow('object_detector', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            processed_image = process_image(frame, detector)
            cv2.imshow('object_detector', processed_image)
            cv2.waitKey(0)  
        elif key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # Turn off all LEDs when the program ends
    red_led.off()
    yellow_led.off()
    green_led.off()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()
    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
        int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
    main()