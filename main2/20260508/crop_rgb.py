from pathlib import Path

from PIL import Image


SRC_ROOT = Path("/home/user/Desktop/main/main2/20260508/rgb_all")
DST_ROOT = Path("/home/user/Desktop/main/main2/20260508/rgb_crop")


def crop_center_half(image: Image.Image) -> Image.Image:
	width, height = image.size
	crop_w = int(width * 0.5)
	crop_h = int(height * 0.5)
	left = (width - crop_w) // 2
	top = (height - crop_h) // 2
	return image.crop((left, top, left + crop_w, top + crop_h))


def main() -> None:
	for src_path in SRC_ROOT.rglob("*"):
		if not src_path.is_file():
			continue

		dst_path = DST_ROOT / src_path.relative_to(SRC_ROOT)
		dst_path.parent.mkdir(parents=True, exist_ok=True)

		with Image.open(src_path) as img:
			crop_center_half(img).save(dst_path)


if __name__ == "__main__":
	main()
