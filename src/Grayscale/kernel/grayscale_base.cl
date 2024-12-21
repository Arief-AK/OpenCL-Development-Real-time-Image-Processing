__kernel void grayscale(
    __global const uchar4* inputBuffer,  // Input buffer (uchar4 per pixel)
    __global uchar4* outputBuffer,       // Output buffer (uchar4 per pixel)
    const int width,                     // Image width
    const int height                     // Image height
) {
    int x = get_global_id(0); // X-coordinate
    int y = get_global_id(1); // Y-coordinate

    // Calculate the 1D index for the current pixel
    int index = y * width + x;

    if (x < width && y < height) {
        uchar4 pixel = inputBuffer[index];

        // Convert uchar4 to normalized float values (0-1 range)
        float r = pixel.x / 255.0f;
        float g = pixel.y / 255.0f;
        float b = pixel.z / 255.0f;

        // Calculate grayscale intensity
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        // Convert grayscale back to uchar (0-255 range) and set alpha to 255
        uchar intensity = (uchar)(gray * 255.0f);
        outputBuffer[index] = (uchar4)(intensity, intensity, intensity, 255);
    }
}
