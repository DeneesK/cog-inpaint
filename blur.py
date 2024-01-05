from PIL import Image, ImageFilter


image = Image.open('output (15).png')
image = image.filter(
    ImageFilter.GaussianBlur(radius=5)
    )
image.show()
image.save('1.png')
