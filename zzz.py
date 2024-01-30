import PIL.Image as Image
import glob

img_list = glob.glob('/data2/coco2017/val2017/*.jpg')

for i in img_list:
    a = Image.open(i)

    print(f"{a.size}")