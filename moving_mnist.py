import numpy as np
import cv2
import os

img_dir = '/home/swlee/data/mnist/training/5/396.png'
img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (100, 100))
print(img.shape)

init_cord = [500, 500]

# for i in range(100):
#     rand_x = np.random.randint(-30, 30)
#     rand_y = np.random.randint(-30, 30)
#     init_cord[0] = init_cord[0]+rand_x
#     init_cord[1] = init_cord[1]+rand_y
#
#     new_img = np.zeros([1000, 1000])
#     new_img[init_cord[0]:init_cord[0]+100, init_cord[1]:init_cord[1]+100] = img
#     cv2.imwrite('test/%d.png' % (i+1), new_img)
#
# os.system('convert -delay 5 -loop 0 {0}/*.png {0}/moving_rand.gif'.format('/home/swlee/projects/Thesis/test/'))

init_cord = [500, 0]
for i in range(9):
    init_cord[1] += 100

    new_img = np.zeros([1000, 1000])
    new_img[init_cord[0]:init_cord[0]+100, init_cord[1]:init_cord[1]+100] = img
    cv2.imwrite('test/%d.png' % (i+1), new_img)

os.system('convert -delay 10 -loop 0 {0}/*.png {0}/moving_hori.gif'.format('/home/swlee/projects/Thesis/test/'))