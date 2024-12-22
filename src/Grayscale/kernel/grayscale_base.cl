__kernel void grayscale(
    __global const uchar4* inputBuffer,  // Input buffer (uchar4 per pixel)
    __global float4* outputBuffer,       // Output buffer (float4 per pixel)
    const int width,                     // Image width
    const int height                     // Image height
) {
    int x = get_global_id(0); // X-coordinate
    int y = get_global_id(1); // Y-coordinate

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel = inputBuffer[index];

        float gray = (0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z) / 255.0f;

        // Output to output image buffer
        outputBuffer[index] = (float4)(gray, gray, gray, 1.0f);
    }
}
