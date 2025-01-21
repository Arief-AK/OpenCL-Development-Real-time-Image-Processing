constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gaussian_blur(__read_only image2d_t input_image,
                            __write_only image2d_t output_image,
                            __constant float *gaussian_kernel,
                            const int kernel_size,
                            int width,
                            int height) {
    
    // Get the position of the current thread
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }
    
    // Initialize the sum of the pixel values
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // Iterate over the kernel
    for (int ky = -kernel_size / 2; ky <= kernel_size / 2; ky++) {
        for (int kx = -kernel_size / 2; kx <= kernel_size / 2; kx++) {
            
            // Get the pixel value and weight
            int2 coord = (int2)(x + kx, y + ky);
            float weight = gaussian_kernel[(ky + kernel_size / 2) * kernel_size + (kx + kernel_size / 2)];
            float4 pixel = read_imagef(input_image, sampler, coord);

            sum += weight * pixel;
        }
    }

    // Write the blurred pixel to the output
    write_imagef(output_image, (int2)(x, y), sum);
}