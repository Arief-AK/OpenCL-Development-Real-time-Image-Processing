__kernel void grayscale(
    __read_only image2d_t input_image,
    __write_only image2d_t output_image,
    const int width,
    const int height)
{
    // Get the index of the current pixel
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Calculate the linear index
    // int idx = y * width + x;

    // Ensure within bounds
    if(x < width && y < height){
        // Coordinates for the pixel
        int2 coord = (int2)(x, y);
        
        // Read RGBA pixel
        float4 pixel = read_imagef(input_image, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, coord);

        // Convert to grayscale using luminance formula
        float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Clamp the value to the range [0.0, 1.0] (normalized)
        gray = clamp(gray, 0.0f, 1.0f);

        // Write image
        write_imagef(output_image, coord, (float4)(gray, gray, gray, 1.0f));
    }
}