import matplotlib.pyplot as plt 
import cv2
import numpy as np

img_BW = cv2.imread('C:/Users/juliesi/Documents/Master/Programmering/sanitycheck_svart-hvit3.JPG', cv2.IMREAD_GRAYSCALE)
# plt.plot(img_BW)
# plt.show()



### Gradients in picture

# compute the gradients along the x and y axis
gX = cv2.Sobel(img_BW, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
gY = cv2.Sobel(img_BW, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)



# compute the gradient magnitude and orientation
magnitude = np.sqrt((gX ** 2) + (gY ** 2))

orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)



# combine the gradient representations into a single image
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)


# show our output images
cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)
cv2.waitKey(0)


txt_path = open('C:/Users/juliesi/Documents/Master/Programmering/asi/txt data fra koden/gradient_sanity_check_mini2', 'w+')

txt_path.write('gX gradient')
for i in range(len(orientation)):
      txt_path.write(str(gX[i]))
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')


txt_path.write('gY gradient')
for i in range(len(orientation)):
    txt_path.write(str(gY[i]))    
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')


txt_path.write('gradient magnitude')
for i in range(len(orientation)):
    # magnitude = np.sqrt((gX[i] ** 2) + (gY[i] ** 2))
    txt_path.write(str(magnitude[i]))
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')
txt_path.write('\n')


txt_path.write('gradient orientation')
for i in range(len(orientation)):
    txt_path.write(str(orientation[i]))
    



txt_path.close()