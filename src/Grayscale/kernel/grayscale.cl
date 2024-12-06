__kernel void grayscale(
    __global const uchar4* input_image,
    __global uchar* output_image,
    const int width,
    const int height)
{
    // Get the index of the current pixel
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Calculate the linear index
    int idx = y * width + x;

    // Ensure within bounds
    if(x < width && y < height){
        // Read RGBA pixel
        uchar4 pixel = input_image[idx];

        // Compute grayscale value using luminance formula
        uchar gray = (uchar)(0.299f * pixel.x + 0.578f * pixel.y + 0.114f * pixel.z);
        output_image[idx] = gray;
    }
}