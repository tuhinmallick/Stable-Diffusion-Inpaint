from PIL import Image, ImageChops
import numpy as np
import cv2 
def are_same_image(image1, image2):
    if np.sum(np.array(ImageChops.difference(image1, image2).getdata())) == 0:
        return True
    return False


# Load the images
img_1_path = "logs/2023-02-08_custom_place_training_different_sampler/images/train/samples_gs-000016_e-000016_b-000000.png"
img_2_path =  "logs/2023-02-08_custom_place_training_different_sampler/images/train/samples_gs-000000_e-000000_b-000000.png"

image1 = Image.open(img_1_path)
image2 = Image.open(img_2_path)

print("METHOD 1", image1 == image2)


# image2 = Image.open("data/INPAINTING/output_images_debug/16693_12.png")
res_numpy = are_same_image(image1,image2)

print("METHOD 2", res_numpy)

a = cv2.imread(img_1_path)
b = cv2.imread(img_2_path)
difference = cv2.subtract(a, b)  
# print(np.any(difference))  
result = not np.any(difference)

print("METHOD 3", result)

