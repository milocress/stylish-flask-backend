import cv2

def slice_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    idx = 0
    framecount = 0
    frame_skip = 10
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret):
            if (idx == frame_skip):
                filename = "test_frames/testframe" + str(framecount) + ".jpg"
                cv2.imwrite(filename, frame)
                framecount += 1
                idx = 0
            else:
                idx += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    slice_frames("cat.mp4")
