import cv2
import numpy as np
import matplotlib as plt

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}
def ImgProcess(roi_color,digits):
    roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    """For viewing the image converted to Gray
    cv2.imshow("gray", roi)
    cv2.waitKey(0)"""

    RATIO = roi.shape[0] * 0.2

    roi = cv2.bilateralFilter(roi, 5, 30, 30)

    trimmed = roi

    edged = cv2.adaptiveThreshold(
        trimmed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    """For viewing the edged image
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    """For viewing the dilated image.
    cv2.imshow("Dilated", dilated)
    cv2.waitKey(0)"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dilated = cv2.dilate(dilated, kernel, iterations=1)

    """For viewing the x2 dilated image.
    cv2.imshow("Dilated x2", dilated)
    cv2.waitKey(0)"""

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1), )
    eroded = cv2.erode(dilated, kernel, iterations=1)
    """For viewing the Eroded image.
    cv2.imshow("Eroded", eroded)
    cv2.waitKey(0)"""

    h = roi.shape[0]
    ratio = int(h * 0.07)
    eroded[-ratio:, ] = 0
    eroded[:, :ratio] = 0
    """For viewing the black eroded image.
    cv2.imshow("Eroded + Black", eroded)
    cv2.waitKey(0)"""

    cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_cnts = []

    canvas = trimmed.copy()
    cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 2)
    """For viewing the all the contours in the image.
    cv2.imshow("All Contours", canvas)
    cv2.waitKey(0)"""

    canvas = trimmed.copy()
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h > 20:
            digits_cnts += [cnt]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.drawContours(canvas, cnt, 0, (255, 255, 255), 1)
            """For viewing the Digit Contours in the image.
            cv2.imshow("Digit Contours", canvas)
            cv2.waitKey(0)"""

    #print(f"No. of Digit Contours: {len(digits_cnts)}")

    """For viewing the Digit Contours in the image.
    cv2.imshow("Digit Contours", canvas)
    cv2.waitKey(0)"""

    sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

    canvas = trimmed.copy()

    for i, cnt in enumerate(sorted_digits):
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.putText(canvas, str(i), (x, y - 3), FONT, 0.3, (0, 0, 0), 1)
    """For viewing the Contours in the image after sorting.
    cv2.imshow("All Contours sorted", canvas)
    cv2.waitKey(0)"""

    #digits = []
    canvas = roi_color.copy()
    for cnt in sorted_digits:
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = eroded[y: y + h, x: x + w]
        #print(f"W:{w}, H:{h}")
        # convenience units
        qW, qH = int(w * 0.25), int(h * 0.15)
        fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)

        # seven segments in the order of wikipedia's illustration
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW, halfH), (w, h)),  # c (lower right)
            ((0, h - qH), (w, h)),  # d (lower bar)
            ((0, halfH), (qW, h)),  # e (lower left)
            ((0, 0), (qW, halfH)),  # f (upper left)
            # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
            (
                (0 + fractionW, halfH - fractionH),
                (w - fractionW, halfH + fractionH),
            ),  # center
        ]

        # initialize to off
        on = [0] * 7

        for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
            region = roi[p1y:p2y, p1x:p2x]
            """
            print(
                f"{i}: Sum of 1: {np.sum(region == 255)}, Sum of 0: {np.sum(region == 0)}, Shape: {region.shape}, Size: {region.size}"
            )"""
            if np.sum(region == 255) > region.size * 0.5:
                on[i] = 1
            #print(f"State of ON: {on}")
        digit = DIGITSDICT[tuple(on)]

        #print(f"Digit is: {digit}")
        digits += [digit]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), CYAN, 1)
        cv2.putText(canvas, str(digit), (x - 5, y + 6), FONT, 0.3, (0, 0, 0), 1)
        """For viewing the Digits with the contours in the image.
        cv2.imshow("Digit", canvas)
        cv2.waitKey(0)"""

    #print(f"Digits on the token are: {digits}")
#Function Ends


digits = []

#Process for obtaining before the decimal points
ImgB4Dcml = cv2.imread("res/demo2.jpg")

""" For viewing the original image for before the decimal points process.
cv2.imshow("original_img", ImgB4Dcml )
cv2.waitKey(0)"""

#for demo2.jpg No. detection before Decimal points
ImgB4Dcml =ImgB4Dcml [310:750,640:1340]

"""For viewing the cropped image before the decimal points.
cv2.imshow("cropped_img", ImgAfterDcml)
cv2.waitKey(0)"""

#Crop size for no. before decimal points
ImgB4Dcml =cv2.resize(ImgB4Dcml  ,(400,70))
ImgProcess(ImgB4Dcml ,digits)

digits += str(".")


#Process for obtaining before the decimal points
ImgAfterDcml = cv2.imread("res/demo2.jpg")
""" For viewing the original image for after the decimal points process.
cv2.imshow("original_img", ImgAfterDcml)
cv2.waitKey(0)"""
ImgAfterDcml=ImgAfterDcml[310:750,1400:1600]
"""For viewing the cropped image after the decimal points.
cv2.imshow("cropped_img", ImgAfterDcml)
cv2.waitKey(0)"""
ImgAfterDcml=cv2.resize(ImgAfterDcml,(100,70))
ImgProcess(ImgAfterDcml,digits)
listToStr=''.join(map(str, digits))
print(listToStr)

