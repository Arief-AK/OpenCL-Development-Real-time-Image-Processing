# Sobel Edge Detection
This document showcases the research and planning of the **Sobel Edge-Detection** image processing method.

## Principle
**Source:** [Improved Sobel Edge Detection](https://ieeexplore.ieee.org/abstract/document/5563693)

The edge of an image is the most basic features of the image. It contains a wealth of internal information of the image. In digital image, the so-called edge is a collection of the pixels whose gray value has a step or roof change, and it also refers to the part where the brightness of the image local area changes significantly. The gray profile in this region can generally be seen as a step. That is, in a small buffer area, **a gray value rapidly changes to another whose gray value is largely different** with it.

**Edge detection is mainly the measurement, detection and location of the changes in image gray**. Therefore, the general method of edge detection is to study the changes of a single image pixel in a gray area, use the variation of the edge neighboring first-order or second-order to detect the edge. This method is used to refer as local operator edge detection method.

The **Sobel–Feldman operator** is based on **convolving the image with a small, separable, and integer-valued filter in the horizontal and vertical directions** and is therefore relatively inexpensive in terms of computations. On the other hand, the gradient approximation that it produces is relatively crude, in particular for high-frequency variations in the image. [source](https://en.wikipedia.org/wiki/Sobel_operator)

Compared to other edge operator, Sobel has two main advantages:
1. **Average factor**: Smoothing effect to random noise of the image
2. **Differrential**: Elements of the edge on both sides are enhanced, **produces thick and bright edges**

## Methodology
The operator uses two **3x3** matrix kernels which are convolved with the original image to calculate aproximations of the derivatives; one for horizontal and another for vertical.

### Base Formula
**Variables**:
- **A**: Source image
- **Gx**: Image matrix with each point contains the **horizontal** derivative approximations
- **Gy**: Image matrix with each point contains the **vertical** derivative approximations

**note**: The `*` operator denotes the 2-D convolution operation

```math
G_x = \begin{bmatrix}+1 & 0 & -1\\+2 & 0 & -2\\+1 & 0 & -1\end{bmatrix} * A \\
G_y = \begin{bmatrix}+1 & +2 & +1\\0 & 0 & 0\\-1 & -2 & -1\end{bmatrix} * A
```

### Improved Formula
The sobel matrix kernels can be decomoposed as products of an **averaging** and a **differentiation** kernel, which computes a gradient with smoothing.

```math
G_x = \begin{bmatrix}1\\2\\1\end{bmatrix} * (\begin{bmatrix}1 & 0 & -1\end{bmatrix} * A)\\
G_y = \begin{bmatrix}+1\\0\\-1\end{bmatrix} * (\begin{bmatrix}1 & 0 & -1\end{bmatrix} * A)
```

In implementations, this separable computation can be advantageous since it implies fewer arithmetic operations for each image point (pixel).

**Properties**:
1. `G_x`: Increasing in the "right-direction"
2. `G_y`: Increasing in the "down-direction"

### Supporting Formulas
At each point (pixel) in the image, the **magnitude** of the gradient can be calculated using:
```math
G = \sqrt{G_x^2 + G_y^2}
```

The **direction** of the gradient can be calculated using:
```math
\theta = atan2(G_y, G_x)
```

Therefore, the complete calculation can be described in pseudocode as:
```math
N(x,y) = \sum_{i=-1}^{1}\sum_{j=-1}^{1}K(i,j)P(x-i,y-j)
```

**Where**:
- **N(x, y)**: New pixel matrix
- **K(i, j)**: `Average` and `Differential` matrices
- **P(x-i, y-j)**: Original matrix

## OpenCL Kernel
TBA

### Parameters
TBA

### Operation
TBA

## Performance Analysis
TBA

### Test Outline
TBA

#### 5 Iterations
TBA

#### 10 Iterations
TBA

#### 100 Iterations
TBA

## Summary
TBA