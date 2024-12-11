import cv2 
import argparse 
parser = argparse.ArgumentParser() 
parser.add_argument("Path", help = "path of the video file", type= str)
parser.add_argument("Background", help = "path of the background image file", type= str )
args = parser.parse_args()

#back = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\Backgroundimage.jpg"
background = cv2.imread(args.Background)
backgr = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(args.Path)

# Create kernel for dilate operation.
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

while True:
    ret, frame = cap.read()
    frame2 = cv2.resize(frame, (256,256))
    frame3 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    fg_mask = cv2.absdiff(frame3, backgr)
   
    _,thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    # Perform dilation to improve the thresholded image.
    
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    #cv2.imshow("mask", thresh)

    # find contours
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    min_contour_area = 100
    large_cont = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_contour_area]
    frame_out = frame2.copy()
    for cont in large_cont:
        x,y,w,h = cv2.boundingRect(cont)
        frame_out = cv2.rectangle(frame2, (x,y), (x+w,y+h), (0,0,255), 3)
        
    cv2.imshow('final_frame', frame_out)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.waitkey(0)
cv2.destroyAllWindows()