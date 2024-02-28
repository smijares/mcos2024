# mcos2023
Code from the S. Mijares, M. Chabert, T. Oberlin and J. Serra-Sagristà paper "Fixed-quality compression of remote sensing images with neural networks" (2024). In this repository you'll find the architecture script, pre-trained models, auxiliary scripts for pre-processing raw images, and usage instructions.

## Abstract

Fixed-quality image compression is a coding paradigm where the tolerated introduced distortion is set by the user. This paper proposes a novel fixed-quality compression method for remote sensing images. It is based on a neural architecture we have recently proposed for multirate satellite image compression. In this paper, we show how to efficiently estimate the reconstruction quality using an appropriate statistical model. The performance of our approach is assessed and compared against recent fixed-quality coding techniques and standards in terms of accuracy and rate-distortion, as well as with recent machine learning compression methods in rate-distortion, showing competitive results. In particular, the proposed method does not introduce artefacts even when coding neighbouring areas at different qualities.

## Contents

This repository is structured as follows:

```
.
├── architecture_uint16.py
├── architecture_uint8.py
├── models
│     ├── 
│     └── 
└── auxiliary
      ├── bands_extractor.py
      └── image_tiler.py

```

## Usage

To run our models, use the architecture scripts. A detailed usage message is printed running the command:

```
python3 architecture_uint16.py -h
```

Our implementation expects images to use a certain naming convention to automatically load the geometry of the image, which is as follows:

```
<name>.<bands>_<width>_<height>_<data type>_<endianness>_<is RGB>.raw
```

The data type is 1 for 8-bit data (unsigned), 2 for 16-bit unsigned integers, 3 for 16-bit signed integers, 4 for 32-bit integers, and 6 32-bit floating point numbers. If an image does not use this naming format, the user must specify these values in their command.

To compress an image using our script, use the `compress` function. For example, the following command is to compress an image using some fixed quality parameter:

```
python3 architecture_uint16.py --model_path ./models/some_model_folder compress /path/to/folder/image.1_512_512_2_1_0.raw --quality 0.001
```

