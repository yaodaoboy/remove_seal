# Remove Seal - Pure Computer Vision Algorithm

This repository contains an implementation of a pure computer vision algorithm designed to remove seals or stamps from images. The algorithm focuses on processing images to detect and eliminate seals while preserving the underlying content.

## Features

- **Seal Detection**: Identifies the location of seals in the image.  
  Note: The `detect_seal_boxes` function in this project may not be sufficiently robust for complex real-world cases. For better results, it is recommended to use a deep-learning-based detector, such as `PaddleOCR`'s seal detection model or a fine-tuned variant.
- **Seal Removal**: Removes the detected seals while maintaining the integrity of the original image.
- **No Machine Learning**: The algorithm is purely based on traditional computer vision techniques, without relying on pre-trained models.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/yaodaoboy/remove_seal.git
	cd remove_seal
	```

2. Install the required dependencies:
	```bash
	pip install -r requirements.txt
	```

## Usage

`remove_seal.py` supports processing a single image file or a whole folder.

1. Run with default paths (`input` -> `output`):
	```bash
	python remove_seal.py
	```

2. Process a single file:
	```bash
	python remove_seal.py --input input/sample.jpg --output output/sample_clean.jpg
	```

3. Process a folder with custom padding:
	```bash
	python remove_seal.py --input input --output output --padding 10
	```

Arguments:

- `--input`: Input image file or folder path. Default: `input`
- `--output`: Output image file or folder path. Default: `output`
- `--padding`: Expansion pixels for detected seal boxes. Default: `8`

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

## Results

The images in the `seals` directory show real seal-removal results, including cases where seals overlap printed text and handwritten text.  
For these processed outputs, you can directly send them to a VLM (Vision-Language Model) to extract the target information.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the algorithm or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This algorithm is designed for educational purposes and may not work perfectly in all scenarios. Use it responsibly and ensure compliance with applicable laws and regulations.
