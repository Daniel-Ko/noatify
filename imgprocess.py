import cv2
import numpy as np

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
	    angle = -(90 + angle)
    else:
        angle= -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))

    cv2.imshow("b4rotation", img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)



def mser(img):

    mser = cv2.MSER_create()

    # vis = img.copy()
    vis = img

    regions, _ = mser.detectRegions(img)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    cv2.imshow('MSER', vis)
    cv2.waitKey(0)

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("text only", text_only)

    return vis

def mask(img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[4]
    cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    cv2.imshow("contours", img)
    cv2.waitKey(0)

def process(img):
    img = cv2.threshold(img, 0, 255, 
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow("THRESH", img)
    # mask(img)
    # threshold = cv2.medianBlur(threshold, 3)
    # select regions
    # mser(img)
    # deskew(img)   
    # mask(img)
    return img