import cv2 as cv
import numpy as np

from pyimagesearch.transform import four_point_transform


def enrode_dilate(img):
    k = np.ones((5, 5), np.uint8)

    # cv.imshow("dilate", dilate)
    # cv.waitKey(0)
    ans = cv.erode(img, k, iterations=10)
    # cv.imshow("erode", ans)
    cv.imwrite("img/erode.jpg", ans)
    ans = cv.dilate(ans, k, iterations=10)
    # cv.imshow("dilate", ans)
    # cv.waitKey(0)
    return ans
def get_counter(mask):

    edges = cv.Canny(mask, 50, 150, apertureSize=3)

    (cnts, _) = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break


    cv.drawContours(img, [screenCnt], -1, (255, 0, 0), 2)
    cv.imshow("Outline", img)
    cv.imwrite("img/outline.jpg", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return screenCnt

img = cv.imread("img/img.jpg")
img1 = img.copy()
Grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite("img/gray.jpg", Grayimg)
ret, thresh = cv.threshold(Grayimg, 245, 255,cv.THRESH_BINARY)
cv.imwrite("img/blackwhite.jpg", thresh)
# cv.imshow('2', thresh)
# cv.waitKey(0)
mask = enrode_dilate(thresh)
cv.imwrite("img/erodedilate.jpg", mask)
# cv.imshow("erode",mask)
# # cv.waitKey(0)
c = cv.Canny(mask, 50, 150, apertureSize=3)
cv.imwrite("img/canny.jpg", c)
# cv.imshow("canny",c)
# cv.waitKey(0)
screenCnt = get_counter(mask)
screenCnt = screenCnt.reshape(4, 2)

warped = four_point_transform(img1, screenCnt.reshape(4, 2))
cv.imwrite("img/warped.jpg", warped)
# cv.imshow("warpped",warped)
cv.waitKey(0)
qiege = warped[266:325,814:1094]
cv.imwrite("img/qiege.jpg", qiege)
