from pathlib import Path

from PIL import Image, ImageDraw


path = Path(__file__).with_name("eeg_bg_studio.ico")
image = Image.new("RGBA", (256, 256), "#0B1118")
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((20, 20, 236, 236), radius=38, fill="#111B24", outline="#263746", width=5)
points = []
for x in range(38, 219):
    y = 128
    if 64 <= x <= 82:
        y -= int((x - 64) * 2.6)
    elif 82 < x <= 100:
        y -= int((100 - x) * 2.6)
    elif 116 <= x <= 126:
        y += int((x - 116) * 4.0)
    elif 126 < x <= 136:
        y += int((136 - x) * 4.0)
    points.append((x, y))
draw.line(points, fill="#21B8A6", width=9, joint="curve")
draw.line([(38, 168), (218, 168)], fill="#4B8DFF", width=7)
image.save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
