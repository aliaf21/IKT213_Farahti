import cv2
import numpy as np

image = cv2.imread("iris.png")

def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    return padded_image

padded_image = padding(image, 100)
cv2.imwrite("padding.png", padded_image)


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    return cropped_image

height, width, channels = image.shape

cropped_image = crop(
    image,
    200,
    width - 130,
    200,
    height - 130
)
cv2.imwrite("crop.png", cropped_image)


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

resized_image = resize(image, 200, 200)
cv2.imwrite("resize.png", resized_image)


emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    return emptyPictureArray

copied_image = copy(image, emptyPictureArray)
cv2.imwrite("copy.png", copied_image)


def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

grayscale_image = grayscale(image)
cv2.imwrite("grayscale.png", grayscale_image)


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

hsv_image = hsv(image)
cv2.imwrite("hsv.png", hsv_image)


emptyHueArray = np.zeros((height, width, 3), dtype=np.uint8)

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_value = int(image[y, x, c]) + hue

                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0

                emptyPictureArray[y, x, c] = new_value

    return emptyPictureArray

hue_image = hue_shifted(image, emptyHueArray, 50)
cv2.imwrite("hue_shifted.png", hue_image)


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)
    return smoothed_image

smoothed_image = smoothing(image)
cv2.imwrite("smoothing.png", smoothed_image)


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated_image = image

    return rotated_image

rotated_image = rotation(image, 180)
cv2.imwrite("rotation_180.png", rotated_image)