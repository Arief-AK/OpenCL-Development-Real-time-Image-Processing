__kernel void sobel_edge_detection(__read_only image2d_t inputImage,
                                   __write_only image2d_t outputImage,
                                   __constant int* sobelX,
                                   __constant int* sobelY,
                                   const int width, const int height){
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int2 coord;
        float gx = 0.0f;
        float gy = 0.0f;

        // Apply Sobel filter
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                coord.x = x + kx;
                coord.y = y + ky;

                float pixel = read_imagef(inputImage, coord).x * 255.0f; // Assuming grayscale input

                gx += pixel * sobelX[(ky + 1) * 3 + (kx + 1)];
                gy += pixel * sobelY[(ky + 1) * 3 + (kx + 1)];
            }
        }

        // Calculate gradient magnitude
        float magnitude = sqrt(gx * gx + gy * gy) / 4.0f; // Normalize to [0, 255]
        magnitude = clamp(magnitude, 0.0f, 255.0f) / 255.0f; // Normalize to [0, 1.0]

        write_imagef(outputImage, (int2)(x, y), (float4)(magnitude, magnitude, magnitude, 1.0f));
    }
}
