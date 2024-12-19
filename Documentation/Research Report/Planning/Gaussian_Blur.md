# Gaussian Blur
This document showcases the research and planning of the **Gaussian Blur** image processing method.

## Principle
**Source**: [Blur clarrified](https://jov.arvojournals.org/article.aspx?articleid=2191790)

In the narrower technical context of vision, optics, and imaging, blurring generally connotes a smearing of an image, through some amount of low-pass filtering. Gaussian blur is defined as convolution with a Gaussian impulse response.

Process of blurring an image by convolving it with a Gauss function is called Gaussian Blurring. It is a method to reduce image noise and details on graphic applications. It involves computing a **weighted average** of neighbouring pixels.[source](https://www.ijml.org/vol5/483-W012.pdf)

## Methodology
Applying a Gaussian blur to images involves the convolution of an image with a Gaussian function. The convolution calculates the transformation to apply to each pixel in the image.

Gaussian function in `1D`:
```math
G(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}
```

Gaussian function in `2D`:
```math
G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
```

**Properties**:
- **x**: Distance from origin in the horizontal axis ((0,0) at centre)
- **y**: Distance from origin in the vertical axis((0,0) at centre)
- **$\sigma$**: Standard deviation of the Gaussian distribution (controls the blur strength)

The values from this function is used to build a convolution matrix which is applied to the original image. Each pixel's new value is set to the **weighted average** of the pixel's neighbourhood (dimension of kernels). The size of the kernel is typically odd (3x3, 5x5, 9x9, .etc).

## OpenCL Kernel
TBA

## Performance Analysis
TBA

## Summary
TBA