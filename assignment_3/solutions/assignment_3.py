import cv2
import numpy as np


def sobel_edge_detection(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    sobel_image = cv2.Sobel(blurred_image, cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_image = cv2.convertScaleAbs(sobel_image)
    cv2.imwrite("sobel_output.png", sobel_image)


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    canny_image = cv2.Canny(blurred_image, threshold_1, threshold_2)
    cv2.imwrite("canny_output.png", canny_image)


def template_match(image, template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    height, width = gray_template.shape

    result = cv2.matchTemplate(
        gray_image,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    locations = np.where(result >= 0.9)

    for point in zip(*locations[::-1]):
        cv2.rectangle(
            image,
            point,
            (point[0] + width, point[1] + height),
            (0, 0, 255),
            2
        )

    cv2.imwrite("template_match_output.png", image)


def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == "up" and scale_factor == 2:
        resized_image = cv2.pyrUp(image)

    elif up_or_down == "down" and scale_factor == 2:
        resized_image = cv2.pyrDown(image)

    else:
        resized_image = image.copy()

    cv2.imwrite(f"resize_{up_or_down}_output.png", resized_image)


lambo = cv2.imread("lambo.png")
shapes = cv2.imread("shapes-1.png")
shapes_template = cv2.imread("shapes_template.jpg")

sobel_edge_detection(lambo)
canny_edge_detection(lambo, 50, 50)
template_match(shapes, shapes_template)
resize(lambo, 2, "up")
resize(lambo, 2, "down")