from PIL import Image, ImageEnhance
import os

image_dir = 'D:/lab/project/deploy/static/upload/123/'
image_list = os.listdir(image_dir)
image = Image.open(image_dir+image_list[0])
converter = ImageEnhance.Color(image)
for i in range(1, 100):
    new_image = converter.enhance(i)
    new_image.save(image_dir + str(i) + '.png')

