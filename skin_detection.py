import cv2
import numpy as np

# Reading the image and converting it into HSV colour space
# Change the argument of the imread function with the path to your desired input image
original_img = cv2.imread('./images/face1.jpeg')
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

# Parameters for the probabilistic model
# Tweakin these parameters will affect the results. Play around with it!
mu0 = [10, 100, 230]
var0 = [500, 5200, 4300]
lmda = 0.2  # P(w=1) = lmda
phi = 0.51  # phi is the threshhold
# Increasing phi reduces false positives and decreasing phi reduces false negatives. A balance must be struck.

# This function will be helpful as we aproximate the pdfs in this model as gaussians


def gaussian(x, mu, var):
    '''
    Returns the value of x for the gaussian distribution with mean=mu and variance=var
    '''
    return (1/np.sqrt(2*np.pi*var))*np.exp(-1*(x-mu)**2/(2*var))

# Gaussian distributions extend out to infinty however colour spaces are finite
# We account for this by computing probability of a given skin pixel being in the entire colour space as per our model
# We use this probability and bayes' theorem to correct our model and ensure we only consider the colours within the colour space
prob_total = [0, 0, 0]
sum = 0
for i in range(0, 180): #For openCV, Hue can vary from 0 to 180
    sum += gaussian(i, mu0[0], var0[0])
prob_total[0] = sum
for d in range(1, 3):
    sum = 0
    for i in range(0, 255): #For openCV, Saturation and value range from 0 to 255
        sum += gaussian(i, mu0[d], var0[d])
    prob_total[d] = sum
# print(prob_total)

def is_skin(x):
    '''
    x should be a 1d-np array of length 3
    It should contain the colour of a pixel in HSV format
    This function uses bayes' theorem to compute the probability of a given pixel being skin
    the computed probability is stored in variable ans
    If probability of pixel being skin exceeds phi, 1 is returned indicating it is a skin pixel
    Otherwise, 0 is returned indicating given pixel is not skin
    '''
    ans =3*gaussian(x[0], mu0[0], var0[0]) / prob_total[0]
    ans += 1*gaussian(x[1], mu0[1], var0[1]) / prob_total[1]
    ans += 1*gaussian(x[2], mu0[2], var0[2]) / prob_total[2]
    ans*=lmda/5
    temp = ans + (1-lmda)*(1/256)
    ans = ans/temp
    if(ans > phi):
        return 1
    else:
        return 0
        
# Simple test for checking parameters 

# Obvious skin pixel. Should return 1.      
# print(is_skin([11.1, 96.9, 237])) 

# Obvious non-skin pixel. Should return 0.
# print(is_skin([30, 30, 50])) 


# Iterating over every pixel in given image and checking if it is a skin-pixel or not
# We store the status of every pixel in a 2d np array called mask
height, width, depth = img.shape
mask = np.zeros((height, width), dtype="int8")
for i in range(height):
    for j in range(width):
        mask[i][j] = is_skin(img[i][j])

# Printing relevant information about given image
print("Dimensions of image:", mask.shape)
print("Number of skin pixels detected:", np.sum(mask))

# We create an image 'skin' that only contains the regions of image detected as skin
skin = cv2.bitwise_and(original_img, original_img, mask = mask)
cv2.imshow('Original Image',original_img) #Displaying original image
cv2.waitKey(0) #Waiting until a key is pressed
cv2.imshow("Detected Skin", skin) #Displaying regions of image detected as skin
cv2.waitKey(0) #Waiting until a key is pressed
cv2.destroyAllWindows() #Closing all image windows

