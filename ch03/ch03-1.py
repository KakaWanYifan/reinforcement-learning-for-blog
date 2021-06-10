import random
from PIL import Image

img = Image.open('TongjiLogo.jpg')
total = 100000
inCount = 0
for i in range(total):
    x = random.randint(0, img.width - 1)
    y = random.randint(0, img.height - 1)
    color = img.getpixel((x, y))
    # 白色的RGB是 (255,255,255)
    if color != (255, 255, 255):
        inCount = inCount + 1

print('蒙特卡洛求校徽面积：')
print(img.width * img.height * inCount / total)
