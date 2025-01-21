__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void sobel_edge_detection(__read_only image2d_t inputImage,
                                   __write_only image2d_t outputImage,
                                   const int width, const int height){
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int2 coord;
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
            {0,  0,  0},
            {1,  2,  1}
        };

        // Apply Sobel filter
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                coord.x = x + kx;
                coord.y = y + ky;

                float pixel = read_imagef(inputImage, sampler, coord).x;

                gx += pixel * sobelX[ky + 1][kx + 1];
                gy += pixel * sobelY[ky + 1][kx + 1];
            }
        }

        // Calculate gradient magnitude
        float magnitude = sqrt(gx * gx + gy * gy);
        magnitude = clamp(magnitude, 0.0f, 1.0f);

        // Write to output image
        write_imagef(outputImage, (int2)(x, y), (float4)(magnitude, magnitude, magnitude, 1.0f));
    }
}
