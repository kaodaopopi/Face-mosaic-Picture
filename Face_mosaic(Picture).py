import cv2

# Load the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load picture
img = cv2.imread('a.jpg')


# Convert to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect face
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(25, 25))
    
# Draw the box of the face part
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#(0, 255, 0)The field can change the box color(Blue,Green,Red)

#Mosaic
def do_mosaic(faces,x,y,w,h,neighbor = 10):
    fh,fw = img.shape[0],img.shape[1]
    if(y+h>fh)or (x+w>fw):
        return
    for i in range(0,h-neighbor,neighbor):
        for j in range(0,w-neighbor,neighbor):
            rect = [j+x,i+y,neighbor,neighbor]
            color = img[i+y][j+x].tolist()
            left_up = (rect[0],rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)
            cv2.rectangle(img, left_up, right_down, color, -1)

do_mosaic(faces,x, y, w, h)

# Show results
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #Normal window size
cv2.imshow('img', img)                     #Show pictures
cv2.imwrite( "result.jpg", img )           #Save Picture
cv2.waitKey(0)                             #Wait for any key to be pressed
cv2.destroyAllWindows()                    #Close window
