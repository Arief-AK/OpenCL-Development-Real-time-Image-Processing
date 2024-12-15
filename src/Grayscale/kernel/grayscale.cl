__kernel void grayscale(
    __read_only image2d_t inputImage,
    __write_only image2d_t outputImage,
    const int width,
    const int height)
{
    int x = get_global_id(0); // X-coordinate
    int y = get_global_id(1); // Y-coordinate

    if (x < get_image_width(inputImage) && y < get_image_height(inputImage)) {
        int2 coord = (int2)(x, y);
        float4 pixel = read_imagef(inputImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, coord);

        // Converts un-normalised values
        float f_x = pixel.x * 255.0f;
        float f_y = pixel.y * 255.0f;
        float f_z = pixel.z * 255.0f;

        printf("Processing pixel: (%d, %d) : %f, %f, %f\n", x, y, f_x, f_y, f_z);
        
        // Grayscale calculation
        float gray = 0.299f * f_x + 0.587f * f_y + 0.114f * f_z;

        // Write the grayscale value to the output image
        write_imagef(outputImage, coord, (float4)(gray, gray, gray, 1.0f));
    }
}