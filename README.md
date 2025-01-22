# OpenCL-Development: Real-time Image Processing
Application of OpenCL for real-time image processing on embedded hardware. This work demonstrates the development of image processing methods such as grayscale conversion, edge-detection, and Gaussian blur. The outcome of this project is to highlight OpenCL's praticality for real-world scenarios.

<p align="center">
    <img alt="Original" src="Documentation/Resources/images/Tulips_medium640.jpg" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
    <img alt="Grayscale" src="Documentation/Resources/images/opencl_grayscale_Tulips_medium640.jpg" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
    <img alt="Gaussian Blur" src="Documentation/Resources/images/opencl_gaussian_Tulips_medium640.jpg" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
    <img alt="Edge-detection" src="Documentation/Resources/images/opencl_edge_Tulips_medium640.jpg" width="20%">
</p>

>[!NOTE]
>
>## Realtime Preview
><details>
> ![Preview gif](Documentation/Resources/images/Realtime%20Preview.gif)
></details>

## OpenCL setup
OpenCL is typically packaged with graphic drivers from vendors like **AMD**, **Intel**, and **NVIDIA**. To ensure that OpenCL is properly installed on your system, install the latest graphic drivers on your device.

- For AMD GPUs, download drivers from the [AMD website](https://www.amd.com/en/resources/support-articles/faqs/GPU-56.html).
- For NVIDIA GPUs, download drivers from the [NVIDIA website](https://www.nvidia.com/en-us/drivers/).
- For Intel GPUs, download drivers from the [Intel website](https://www.intel.com/content/www/us/en/download-center/home.html).

### Linux
On Linux machines, it is recommended to install the `ocl-icd-opencl-dev` package
```shell
sudo apt-get install ocl-icd-opencl-dev
```

## OpenCV setup
All applications in this project utilises the OpenCV library. See the official [OpenCV website](https://opencv.org/get-started/).

### Windows
Download the package from the official [OpenCV website](https://opencv.org/get-started/) or its [GitHub page](https://github.com/opencv/opencv/releases/latest).

### Linux
On Linux machines, run the following command
```shell
sudo apt install libopencv-dev clinfo
```

## Setup

> [!NOTE]\
> Ensure that `CMake` is properly installed and added to the `PATH` environment variable on your machine. See [CMake official documentation](https://cmake.org/download/).
>
>This project is developed using [visual studio code](https://code.visualstudio.com/) IDE. The [CMake Tools extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) is used extensively within the project development. It is highly recommended that users develop with VS code with this extension.