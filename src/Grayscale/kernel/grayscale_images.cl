__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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

        float4 pixel = read_imagef(inputImage, sampler, (int2) (x,y));
        
        // Grayscale calculation
        float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Write the grayscale value to the output image
        write_imagef(outputImage, (int2)(x, y), (float4)(gray, 0.0f, 0.0f, 1.0f));
    }
}