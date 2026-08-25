import cv2

image = cv2.imread("iris-1.jpg")

def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)


def main():
    print_image_information(image)


if __name__ == "__main__":
    main()

    import cv2

    camera = cv2.VideoCapture(0)

    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open("camera_outputs.txt", "w") as file:
        file.write(f"FPS: {fps}\n")
        file.write(f"Height: {height}\n")
        file.write(f"Width: {width}\n")

    camera.release()