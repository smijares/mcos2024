# mcos2024
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

To compress at a fixed quality, first the prediction coefficients need to be instantiated. While some predefined ones are available with the released models, it may be recommended to generate new ones for the specific dataset to be used to ensure better precision. These coefficients may be generated using the command:

```
python3 architecture_uint16.py --model_path ./models/some_model_folder generate_coefficients "/path/to/folder/*raw" --minimum_quality 0.00001 --maximum_quality 0.01
```

With the prediction coefficients at hand, fixed-quality compression can be run using the command:

```
python3 architecture_uint16.py --model_path ./models/some_model_folder FQcompress /path/to/folder/image.1_512_512_2_1_0.raw <target MSE> <patchsize>
```

If the target MSE is too high or too low for the model, then the minimum or maximum quality parameters specified in the coefficients generation will be used, which may lead to results significantly different from the specified target. The patchsize must be a multiple of 16, as that is the minimum resolution provided in the latent space due to downsampling in this architecture.

To decompress an image, regardless of the compression function that has been used, a command such as the following will work:

```
python3 architecture_uint16.py --model_path ./models/some_model_folder decompress /path/to/folder/image.1_512_512_2_1_0.raw.tfci
```

## Data sets

The public data sets used in the paper to reproduce our results is available in the following repositories.

* [Landsat 8 OLI](https://gici.uab.cat/GiciWebPage/datasets.php)
* [AVIRIS calibrated](https://gici.uab.cat/GiciWebPage/datasets.php)
* [AVIRIS uncalibrated](https://cwe.ccsds.org/sls/docs/Forms/AllItems.aspx?RootFolder=%2Fsls%2Fdocs%2FSLS%2DDC%2F123%2E0%2DB%2DInfo%2FTestData%2FAVIRIS&FolderCTID=0x012000439B56FF51847E41B5728F9730D7B55F&View=%7BAE8FB44C%2DE80A%2D42CF%2D8558%2DFB495ABB675F%7D)
