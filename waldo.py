'''
Yitzak Hernandez
UCF
'''

# import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os   

waldoCount = 1

def main():
    images = ['images/puzzle_1.jpg', 'images/puzzle_2.png']
    queries = ['images/query_1.jpg', 'images/query_2.png']

    if not os.path.exists('images_results'):
        os.makedirs('images_results')

    i = 0
    while i < len(images):
        print("Processing Image: " + str(waldoCount))
        findWaldo(images[i], queries[i])
        i += 1


# Perform template matching: Slide template image over scene image and
# get scores for matches at each position and display final result
def findWaldo(image, query):
    count = 0
    global waldoCount
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    # Load puzzle image
    im = cv2.imread(image)
    imcopy = im.copy()
    
    # Load query image
    imq = cv2.imread(query)

    # Get the dimensions of Waldo's image
    waldoH, waldoW = imq.shape[:2]

    for meth in methods:
        # Make a copy of puzzle image
        im = imcopy.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(im,imq,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + waldoH, top_left[1] + waldoH)

        # Draw rectangle on image where the best score is found
        cv2.rectangle(im,top_left, bottom_right, 255, 2)

        # Display the results using Matplotlib and save figure in file
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(im,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        # Display image using OpenCV
        plt.show()
    
        # Save in file using OpenCV
        cv2.imwrite('images_results/result' + str(waldoCount) + '.' + methods[count] + '.jpg', im)
        count += 1
    pass

    print("Completed Processing Image: " + str(waldoCount))
    waldoCount += 1


if __name__ == "__main__":
    main()