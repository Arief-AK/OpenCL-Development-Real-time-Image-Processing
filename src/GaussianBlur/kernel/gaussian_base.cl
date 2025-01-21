__kernel void gaussian_blur(
    __global const uchar4* input_buffer,  // Input buffer (uchar4 per pixel)
    __global uchar4* output_buffer,       // Output buffer (uchar4 per pixel)
    __constant float* gaussian_kernel,    // Gaussian kernel
    const int kernel_size,                // Kernel size (e.g., 3, 5, 7)
    const int width,                      // Image width
    const int height                      // Image height
) {
    // Get the position of the current thread
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Ensure the thread is within the image bounds
    if (x >= width || y >= height) {
        return;
    }

    // Initialize the sum of pixel values and weights
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float total_weight = 0.0f;

    // Iterate over the kernel window
    for (int ky = -kernel_size / 2; ky <= kernel_size / 2; ky++) {
        for (int kx = -kernel_size / 2; kx <= kernel_size / 2; kx++) {
            // Compute the neighboring pixel coordinates
            int neighbor_x = x + kx;
            int neighbor_y = y + ky;

            // Clamp coordinates to image boundaries
            neighbor_x = clamp(neighbor_x, 0, width - 1);
            neighbor_y = clamp(neighbor_y, 0, height - 1);

            // Compute the linear index of the neighbor
            int neighbor_index = neighbor_y * width + neighbor_x;

            // Get the pixel and kernel weight
            uchar4 pixel = input_buffer[neighbor_index];
            float weight = gaussian_kernel[(ky + kernel_size / 2) * kernel_size + (kx + kernel_size / 2)];

            // Accumulate the weighted pixel value
            sum += convert_float4(pixel) * weight;
            total_weight += weight;
        }
    }

    // Normalize the result and write to the output buffer
    int index = y * width + x;
    sum /= total_weight; // Normalize using the total weight
    output_buffer[index] = convert_uchar4(sum);
}
