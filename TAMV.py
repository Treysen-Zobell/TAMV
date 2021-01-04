
# Error codes
# [0] No error
# [1] CV2 ImportError
# [2] DuetWebAPI ImportError
# [3] SSH Not Permitted Error
# [4] Printer is not a Duet V2 or V3
# [5]

# Imports
import numpy as np
import threading
import argparse
import datetime
import imutils
import queue
import time
import sys
import os

try:
    import cv2
except ImportError:
    print('Import CV2 failed. Please install openCV.')
    print('Suggested: https://github.com/DanalEstes/PiInstallOpenCV')
    sys.exit(1)

try:
    import DuetWebAPI
except ImportError:
    print('Python library module "DuetWebAPI.py" is required.')
    print('Obtain from: https://github.com/DanalEstes/DuetWebAPI')
    print('Place in same directory as script, or in Python libpath')
    sys.exit(2)


class ThreadCommunication:
    # Consts
    stop = [0]
    ready = [1]
    frame_data = [2, [0, 0], [0, 0]]
    message_command = [3, '']
    extra_text = [4, '']
    crosshair = [5, 0]
    rotate = [6]
    reset_rotation = [7]
    exit = [8]

    # Shared members
    txq = None
    rxq = None
    printer = None
    cp_coords = None
    tool_coords = None
    camera = None
    repeat = None

    def __init__(self):
        pass

    @staticmethod
    def init():
        ThreadCommunication.rxq = queue.SimpleQueue()
        ThreadCommunication.txq = queue.SimpleQueue()
        ThreadCommunication.txq.put(ThreadCommunication.stop)


def main():
    # Environment checks
    if os.environ.get('SSH_CLIENT'):
        print('This script MUST be run on the graphics console, not an SSH session')
        sys.exit(3)
    os.environ['QT_LOGGING_RULES'] = 'qt5ct.debug=false'

    # Read arguments
    parser = argparse.ArgumentParser(description='Program to allign multiple tools on Duet based printers, using machine vision.', allow_abbrev=False)
    parser.add_argument('-duet', type=str, nargs=1, default=['localhost'],
                        help='Name or IP address of Duet printer. You can use -duet=localhost if you are on the embedded Pi on a Duet3.')
    parser.add_argument('-vidonly', action='store_true', help='Open video window and do nothing else.')
    parser.add_argument('-camera', type=int, nargs=1, default=[0], help='Index of /dev/videoN device to be used.  Default 0. ')
    parser.add_argument('-cp', type=float, nargs=2, default=[0.0, 0.0], help="x y that will put 'controlled point' on carriage over camera.")
    parser.add_argument('-repeat', type=int, nargs=1, default=[1], help="Repeat entire alignment N times and report statistics")
    args = vars(parser.parse_args())

    duet = args['duet'][0]
    video_only = args['vidonly']
    ThreadCommunication.camera = args['camera'][0]
    control_point = args['cp']
    ThreadCommunication.repeat = args['repeat'][0]

    # Video stream
    video_stream_thread = threading.Thread(target=run_video_stream)
    video_stream_thread.start()

    if video_only:
        video_window()

    # Connect to printer
    print(f'Attempting to connect to printer at {duet}')
    ThreadCommunication.printer = DuetWebAPI.DuetWebAPI(f'http://{duet}')

    if not ThreadCommunication.printer.printerType():
        print(f'Device at {duet} either did not respond or is not a Duet V2 or V3 printer')
        sys.exit(4)

    # User info
    print('')
    print('#########################################################################')
    print('# Important:                                                            #')
    print('# Your printer MUST be capable of mounting and parking every tool with  #')
    print('# no collisions between tools.                                          #')
    print('#                                                                       #')
    print('# Offsets for each tool must be set roughly correctly, even before the  #')
    print('# first run of TAMV. They do not need to be perfect; only good enough   #')
    print('# to get the tool on camera. TAMV has to see it to align it.           #')
    print('#########################################################################')
    print('')
    print('#########################################################################')
    print('# Hints:                                                                #')
    print('# Preferred location for the camera is along the max Y edge of the bed. #')
    print('# Fixed camera is also OK, but may limit aligning tools that differ     #')
    print('# significantly in Z.                                                   #')
    print('#                                                                       #')
    print('# If circle detect finds no circles, try changing lighting, changing z, #')
    print('# cleaning the nozzle, or slightly blurring focus.                      #')
    print('#                                                                       #')
    print('# Quite often, the open end of the insulation on the heater wires will  #')
    print('# be detected as a circle.  They may have to be covered.                #')
    print('#                                                                       #')
    print('# Your "controlled point" can be anything. Tool changer pin is fine.    #')
    print('#########################################################################')
    print('')

    if control_point[1] == 0:
        controlled_point()
    else:
        control_point_coords = {'X': control_point[0], 'Y': control_point[1]}

    tool_coords = []
    for i in range(ThreadCommunication.repeat):
        tool_coords.append([])
        for j in range(ThreadCommunication.printer.getNumTools()):
            tool_coords[i].append(each_tool(j, i))

    print('Unmounting last tool')
    ThreadCommunication.printer.gCode('T-1')

    print()
    for i in range(len(tool_coords[0])):
        tool_offsets = ThreadCommunication.printer.getG10ToolOffset(i)
        x = np.around((control_point_coords['X'] + tool_offsets['X']) - tool_coords[0][i]['X'], 3)
        y = np.around((control_point_coords['Y'] + tool_offsets['Y']) - tool_coords[0][i]['Y'], 3)
        print(f'G10 P{i} X{x} Y{y}')
    print()

    if ThreadCommunication.repeat > 1:
        repeat_report()

    ThreadCommunication.txq.put([ThreadCommunication.exit])

    print('')
    print('If your camera is in a consistent location, next time you run TAMV, ')
    print(f'you can optionally supply -cp {control_point_coords["X"]} {control_point_coords["X"]}')
    print('Adding this will cause TAMV to skip all interaction, and attempt to align all tools on its own.')
    print('(This is really the x y of your camera)')


def video_window():
    # User info
    print('')
    print('Video Window only selected with -vidonly')
    print('Press enter to toggle crosshair vs circle finder.')
    print('Press Ctrl+C to exit.')

    # Thread communication
    ThreadCommunication.txq.put([ThreadCommunication.stop])
    ThreadCommunication.txq.put([ThreadCommunication.crosshair, True])
    ThreadCommunication.txq.put([ThreadCommunication.reset_rotation])

    # Main loop
    try:
        toggle = True
        while True:
            input()
            toggle = not toggle
            ThreadCommunication.txq.put([ThreadCommunication.crosshair, toggle])
    except KeyboardInterrupt:
        ThreadCommunication.txq.put(ThreadCommunication.exit)
        exit(0)


def create_detector(t1=20, t2=200, all=0.5, area=200):
    # Setup cv2.SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = t1  # Change thresholds
    params.maxThreshold = t2
    params.filterByArea = True  # Filter by Area.
    params.minArea = area
    params.filterByCircularity = True  # Filter by Circularity
    params.minCircularity = all
    params.filterByConvexity = True  # Filter by Convexity
    params.minConvexity = all
    params.filterByInertia = True  # Filter by Inertia
    params.minInertiaRatio = all
    # ver = (cv2.__version__).split('.') # Create a detector with the parameters
    # if int(ver[0]) < 3 :
    #    detector = cv2.SimpleBlobDetector(params)
    # else:
    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def vector_distance(xy1, xy2):
    # Final rounding and into to convert to pixel values
    return int(np.around(np.sqrt(
        (xy2[0] - xy1[0]) ** 2 +
        (xy2[1] - xy1[1]) ** 2
    )))


def print_keypoint_xyr(keypoints):
    for i in range(len(keypoints)):
        print(f'Keypoint {i} XY = {np.around(keypoints[i].py, 3)}')
        print(f'Keypoints {i} R = {np.around(keypoints[i].size / 2, 3)}')


def controlled_point():
    # Thread communication
    ThreadCommunication.printer.gCode('T-1')
    ThreadCommunication.txq.put([ThreadCommunication.stop])
    ThreadCommunication.txq.put([ThreadCommunication.crosshair, True])
    ThreadCommunication.txq.put([ThreadCommunication.reset_rotation])

    # User info
    print('#########################################################################')
    print('# 1) Using Duet Web, jog until your controlled point appears.           #')
    print('# 2) Using Duet Web, very roughly center the controled point            #')
    print('# 3) Click back in this script window, and press Ctrl+C                 #')
    print('#########################################################################')

    # Main loop
    try:
        while True:
            message = input('Enter message to send to subthread')
            ThreadCommunication.txq.put([ThreadCommunication.message_command, message])
    except KeyboardInterrupt:
        print()
        print('Capturing raw position of the control point')
        ThreadCommunication.cp_coords = ThreadCommunication.printer.getCoords()
        print(f'Controlled point X{ThreadCommunication.cp_coords["X"]} Y{ThreadCommunication.cp_coords["Y"]}')
        ThreadCommunication.txq.put([ThreadCommunication.crosshair, False])
        return None


def each_tool(tool, rep):
    # Thread communication
    ThreadCommunication.txq.put([ThreadCommunication.stop])
    ThreadCommunication.txq.put([ThreadCommunication.crosshair, False])

    # IDK, someone who understands this comment it
    average = [0, 0]
    guess = [1, 1]  # Millimeters
    # target = [720 / 2, 480 / 2]  # Pixels (Will be recalculated)
    direction = [-1, -1]  # X+ or X-, Y+ or Y- Determined through initial move
    xy = [0, 0]
    old_xy = xy
    state = 0  # Machine state for solving image rotation
    # rotation = 0  # Rotation of image in degrees
    count = 0

    # User info
    print('')
    print('')
    print(f'Mounting tool T{tool} for repeat pass {rep + 1}')

    # Printer GCode
    ThreadCommunication.printer.gCode(f'T{tool}')
    ThreadCommunication.printer.gCode(f'G1 X{np.around(ThreadCommunication.cp_coords["X"], 3)} F5000')
    ThreadCommunication.printer.gCode(f'G1 Y{np.around(ThreadCommunication.cp_coords["Y"], 3)} F5000')

    # Ignore any frame message that occurred before this point
    while not ThreadCommunication.rxq.empty():
        ThreadCommunication.rxq.get()
    ThreadCommunication.txq.put([ThreadCommunication.ready])

    # User info on first tool
    if tool == 0:
        print('#########################################################################')
        print('# If tool does not appear in window, adjust G10 Tool offsets to be      #')
        print('# roughly correct.  Then re-run TAMV from the beginning.                #')
        print('#                                                                       #')
        print('# If no circles are found, try slight jogs in Z, changing lighting,     #')
        print('# and cleaning the nozzle.                                              #')
        print('#########################################################################')

    # Main loop
    while True:
        # Check if incoming message is present
        if ThreadCommunication.rxq.empty():
            ThreadCommunication.txq.put([ThreadCommunication.ready])
            time.sleep(0.1)
            continue

        # Read next message in queue
        queue_message = ThreadCommunication.rxq.get()
        if queue_message[0] != ThreadCommunication.frame_data:
            print(f'Skipping unknown queue message header {queue_message[0]}')
            continue

        # Found one circle, process it.
        xy = queue_message[1]
        target = queue_message[2]

        # Keep track of center of circle and average across many circles
        average[0] += xy[0]
        average[1] += xy[1]
        count += 1

        if count > 15:
            average[0] /= count
            average[1] /= count
            average = np.around(average, 3)

            if state == 0:
                print('Initiating a small X move to calibrate camera to carriage rotation')
                old_xy = xy

                # Printer GCode
                ThreadCommunication.printer.gCode('G91 G1 X-0.5 G90')

                # Clear message queue
                while not ThreadCommunication.rxq.empty():
                    ThreadCommunication.rxq.get()
                ThreadCommunication.txq.put([ThreadCommunication.ready])

                state += 1

            elif state == 1:
                if abs(int(old_xy[0] - xy[0]) > int(2 + abs(old_xy[1] - xy[1]))):
                    print('Found X movement via rotation, will now calibrate camera to carriage direction')
                    mm_per_pixel = 0.5 / vector_distance(xy, old_xy)
                    print(f'MM per Pixel distance = {mm_per_pixel}')
                    pixel_per_mm = vector_distance(xy, old_xy) / 0.5
                    print(f'Pixel per MM distance = {pixel_per_mm}')
                    state += 1
                    old_xy = xy
                    direction = [1, 1]

                else:
                    print('Camera to carriage movement axis incompatible... will rotate image and calibrate again.')
                    ThreadCommunication.txq.put([ThreadCommunication.stop])
                    ThreadCommunication.txq.put([ThreadCommunication.rotate])
                    state = 0

            elif state == 2:
                for i in [0, 1]:
                    if abs(target[i] - old_xy[i]) < abs(target[i] - xy[i]):
                        print('Detected movement away from target, now reversing')
                        direction[i] *= -1

                    guess[i] = np.around((target[i] - xy[i]) / (pixel_per_mm * 2), 3)
                    guess[i] *= direction[i]

                ThreadCommunication.printer.gCode(f'G91 G1 X{guess[0]} Y{guess[1]} G90')
                print(f'G91 G1 X{guess[0]} Y{guess[1]} G90')
                old_xy = xy

                if np.around(guess[0], 3) == 0 and np.around(guess[1], 3) == 0:
                    ThreadCommunication.txq.put([ThreadCommunication.stop])
                    print(f'Found center of image at offset coordinates {ThreadCommunication.printer.getCoords()}')
                    center = ThreadCommunication.printer.getCoords()
                    center['mm_per_pixel'] = mm_per_pixel
                    return center

            average = [0, 0]
            count = 0


# Print current repeatability statistics for
def repeat_report():
    print()
    print(f'Repeatability statistics for {ThreadCommunication.repeat} repeats:')
    print('+-------------------------------------------------------------------------------------------+')
    print('|   |                           X                   |                   Y                   |')
    print('| T |  MPP  |   Avg   |   Max   |   Min   |  StdDev |   Avg   |   Max   |   Min   |  StdDev |')
    for t in range(ThreadCommunication.getNumTools()):
        #      | 0 | 123 |123.456 | 123.456 | 123.456 | 123.456 | 123.456 | 123.456 | 123.456 | 123.456 |
        # This printing style is bulky, should be revised to make the code more readable
        print('| {0:1.0f} '.format(t), end='')
        print('| {0:3.3f} '.format(np.around(np.average([ThreadCommunication.tool_coords[i][t]['MPP'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.average([ThreadCommunication.tool_coords[i][t]['X'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.max([ThreadCommunication.tool_coords[i][t]['X'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.min([ThreadCommunication.tool_coords[i][t]['X'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.std([ThreadCommunication.tool_coords[i][t]['X'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.average([ThreadCommunication.tool_coords[i][t]['Y'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.max([ThreadCommunication.tool_coords[i][t]['Y'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.min([ThreadCommunication.tool_coords[i][t]['Y'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('| {0:7.3f} '.format(np.around(np.std([ThreadCommunication.tool_coords[i][t]['Y'] for i in range(ThreadCommunication.repeat)]), 3)), end='')
        print('|')
    print('+-------------------------------------------------------------------------------------------+')
    print('Note: Repeatability cannot be better than one pixel, see Millimeters per Pixel, above.')


# Draw text on image
def put_text(frame, text, color=(0, 0, 255), offsetx=0, offsety=0, stroke=1):
    if text == 'timestamp':
        text = datetime.datetime.now().strftime('%m-%d-%Y $H:%M:%S')

    baseline = 0
    font_scale = 1

    if frame.shape[1] > 640:
        font_scale = 2
        stroke = 2

    offset_pixels = cv2.getTextSize('A', cv2.FONT_HERSHEY_SIMPLEX, font_scale, stroke)
    text_pixels = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, stroke)
    offsetx = max(offsetx, (-frame.shape[1] / 2 + offset_pixels[0][0]) / offset_pixels[0][0])
    offsetx = min(offsetx, (frame.shape[1] / 2 - offset_pixels[0][0]) / offset_pixels[0][0])
    offsety = max(offsety, (-frame.shape[0] / 2 + offset_pixels[0][1]) / offset_pixels[0][1])
    offsety = min(offsetx, (frame.shape[0] / 2 - offset_pixels[0][1]) / offset_pixels[0][1])
    cv2.putText(frame, text,
                int(offsetx * offset_pixels[0][0]) + int(frame.shape[1] / 2) - int(text_pixels[0][0] / 2),
                int(offsety * offset_pixels[0][1]) + int(frame.shape[0] / 2) - int(text_pixels[0][1] / 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, stroke)
    return frame


# Runs in separate thread to own camera
def run_video_stream():
    # q = None
    rotation = 0
    xy = [0, 0]
    # old_xy = xy
    # state = 0
    # count = 0
    rd = 0
    # queue_message = [0, '']
    # extra_text = ''
    mono = False
    blur = [False, 0]
    okay_to_send = 0
    crosshair = 0
    no_circle = 0

    detector = create_detector()
    video_stream = cv2.VideoCapture(ThreadCommunication.camera)

    while True:
        # Process queue messages before frames
        if not ThreadCommunication.txq.empty():
            queue_message = ThreadCommunication.txq.get()
            if queue_message[0] == ThreadCommunication.exit:
                return 0
            elif queue_message[0] == ThreadCommunication.stop:
                okay_to_send = 0
            elif queue_message[0] == ThreadCommunication.ready:
                okay_to_send = 1
            elif queue_message[0] == ThreadCommunication.crosshair:
                crosshair = queue_message[1]
            elif queue_message[0] == ThreadCommunication.extra_text:
                extra_text = queue_message[1]
            elif queue_message[0] == ThreadCommunication.rotate:
                rotation = (rotation + 90) % 360  # Add 90deg but keep wrap around to keep between 0-360
            elif queue_message[0] == ThreadCommunication.reset_rotation:
                rotation = 0
            elif queue_message[0] == ThreadCommunication.message_command:
                try:
                    if 'mono' in queue_message[1]:
                        mono = not mono
                    if 'blur' in queue_message[1]:
                        blur = [not blur, int(queue_message[1].split()[1])]
                    if 'thresh' in queue_message[1]:
                        detector = create_detector(t1=int(queue_message[1].split[1]), t2=int(queue_message[1].split()[2]))
                    if 'all' in queue_message[1]:
                        detector = create_detector(all=float(queue_message[1].split()[1]))
                    if 'area' in queue_message[1]:
                        detector = create_detector(area=int(queue_message[1].split()[1]))

                # Should handle possible exceptions individually, Exception is too broad
                except Exception as e:
                    print('Bad command or argument')

        # Process frames
        (success, cap_frame) = video_stream.read()
        frame = imutils.rotate_bound(cap_frame, rotation)
        target = [int(np.around(frame.shape[1] / 2)), int(np.around(frame.shape[0] / 2))]

        if mono:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur[0]:
            frame = cv2.medianBlur(frame, 'Q')

        keypoints = detector.detect(frame)

        # Draw timestamp on frame after circle detection to prevent finding circles in text
        frame = put_text(frame, 'timestamp', offsety=99)
        frame = put_text(frame, 'Q', offsetx=99, offsety=-99)
        if not okay_to_send:
            frame = put_text(frame, '-', offsetx=99, offsety=-99)

        if crosshair:
            frame = cv2.line(frame, (target[0], target[1] - 25), (target[0], target[1] + 25), (0, 255, 0), 1)
            frame = cv2.line(frame, (target[0] - 25, target[1]), (target[0] - 25, target[1]), (0, 255, 0), 1)
            cv2.imshow('Nozzle', frame)
            key = cv2.waitKey(1)
            continue

        if no_circle > 25:
            show_blobs(cap_frame)
            no_circle = 0

        number_of_circles = len(keypoints)
        # Found no circles
        if number_of_circles == 0:
            if 25 < int(round(time.time() * 1000)) - rd:
                no_circle += 1
                frame = put_text(frame, 'No circles found', offsety=3)
                cv2.imshow('Nozzle', frame)
                cv2.waitKey(1)
            continue

        # Found too many circles (>1)
        elif number_of_circles > 1:
            if 25 < int(round(time.time() * 1000)) - rd:
                frame = put_text(frame, f'Too many circles found {number_of_circles}', offsety=3, color=(255, 255, 255))
                frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("Nozzle", frame)
                cv2.waitKey(1)
            continue

        # Found 1 circle
        no_circle = 0
        center_position = np.uint16(np.around(keypoints[0].pt))
        center_radius = np.around(keypoints[0].size / 2)
        frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = put_text(frame, f'X{center_position[0]} Y{center_position[1]} R{center_radius}', offsety=2, color=(0, 255, 0), stroke=2)

        cv2.imshow('Nozzle', frame)
        cv2.waitKey(1)

        rd = int(round(time.time() * 1000))

        if okay_to_send:
            ThreadCommunication.rxq.put([ThreadCommunication.frame_data, center_position, target])


def show_blobs(image):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True  # Filter by Area.
    params.minArea = 150
    params.filterByCircularity = False  # Filter by Circularity
    params.filterByConvexity = False  # Filter by Convexity
    params.filterByInertia = False  # Filter by Inertia
    params.minInertiaRatio = 0.15

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)

    frame = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame = put_text(frame, 'timestamp', offsety=99)
    frame = put_text(frame, 'Blobs with fewer filters', offsety=4)

    cv2.imshow('Blobs', frame)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
