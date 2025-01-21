__kernel void sobel_edge_detection(
    __global const uchar4* input_buffer,  // Input buffer (uchar4 per pixel)
    __global float* output_buffer,        // Output buffer (grayscale float per pixel)
    const int width,                      // Image width
    const int height                      // Image height
) {
    // Get the position of the current thread
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure the thread is within bounds
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f;
        float gy = 0.0f;

        // Sobel Kernels
        const int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int sobelY[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        // Apply Sobel filter
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                // Compute the neighboring pixel coordinates
                int neighbor_x = x + kx;
                int neighbor_y = y + ky;

                // Compute the linear index of the neighbor
                int neighbor_index = neighbor_y * width + neighbor_x;

                // Read the grayscale value of the neighbor pixel
                uchar4 neighbor_pixel = input_buffer[neighbor_index];
                float gray = (0.299f * neighbor_pixel.x + 0.587f * neighbor_pixel.y + 0.114f * neighbor_pixel.z) / 255.0f;

                // Accumulate Sobel filter responses
                gx += gray * sobelX[ky + 1][kx + 1];
                gy += gray * sobelY[ky + 1][kx + 1];
            }
        }

        // Compute the gradient magnitude
        float magnitude = sqrt(gx * gx + gy * gy);
        magnitude = clamp(magnitude, 0.0f, 1.0f); // Ensure the result is within [0, 1]

        // Write the result to the output buffer
        int index = y * width + x;
        output_buffer[index] = magnitude;
    }
}
