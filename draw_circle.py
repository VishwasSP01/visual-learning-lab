from PIL import Image, ImageDraw

image = Image.new("RGB", (64, 64), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((16, 16, 48, 48), fill="blue")

image.save("blue_circle.png")
