#include <stdio.h>
#include <cstdlib>
#include "arrayfire.h"

using namespace af;

int main(int argc, char** argv)
{
    /* Harris Corner Detector */
    // Corner detector based off of the ArrayFire tutorials
    // Using this as a base before moving onto a full blown
    // stereo-block matcher!

	char t_device_name[64] = {0};
	char t_device_platform[64] = {0};
	char t_device_toolkit[64] = {0};
	char t_device_compute[64] = {0};
	af::deviceInfo(t_device_name, t_device_platform, t_device_toolkit, t_device_compute);

    printf("Device name: %s\n", t_device_name);
    printf("Platform name: %s\n", t_device_platform);
    printf("Toolkit: %s\n", t_device_toolkit);
    printf("Compute version: %s\n", t_device_compute);

    std::cout << "Beginning Work: " << std::endl;
    array img_color = loadimage("start_gate.png", true);
    array img_gray = colorspace(img_color, AF_GRAY, AF_RGB);
    img_color /= 255.f;

    array ix, iy;
    grad(ix, iy, img_gray);

    // Compute second-order derivatives
    array ixx = ix * ix;
    array ixy = ix * iy;
    array iyy = iy * iy;

    // Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
    // These values can be changed to use a smaller or larger window
    array gauss_filt = gaussiankernel(5, 5, 1.0, 1.0);

    // Filter second-order derivatives with Gaussian kernel computed previously
    ixx = convolve(ixx, gauss_filt);
    ixy = convolve(ixy, gauss_filt);
    iyy = convolve(iyy, gauss_filt);

    // Calculate trace
    array tr = ixx + iyy;
    // Calculate determinant
    array det = ixx * iyy - ixy * ixy;

    // Calculate Harris response
    array response = det - 0.04f * (tr * tr);
    // Gets maximum response for each 3x3 neighborhood
    array max_resp = maxfilt(response, 3, 3);

    // Discard responses that are not greater than threshold
    array corners = response > 1e5f;
    corners = corners * response;

    // Discard responses that are not equal to maximum neighborhood response,
    // scale them to original response value
    corners = (corners == max_resp) * corners;
    unsigned good_corners = 0;

    // Gets host pointer to response data
    float* h_corners = corners.host<float>();

    // Draw draw_len x draw_len crosshairs where the corners are
    // This is CPU code!
    const int draw_len = 3;
    for (int y = draw_len; y < img_color.dims(0) - draw_len; y++) {
        for (int x = draw_len; x < img_color.dims(1) - draw_len; x++) {
            // Only draws crosshair if is a corner
            if (h_corners[x * corners.dims(0) + y] > 1e5f) {
                // Draw horizontal line of (draw_len * 2 + 1) pixels centered on the corner
                // Set only the first channel to 1 (green lines)
                img_color(y, seq(x-draw_len, x+draw_len), 0) = 0.f;
                img_color(y, seq(x-draw_len, x+draw_len), 1) = 1.f;
                img_color(y, seq(x-draw_len, x+draw_len), 2) = 0.f;

                // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
                // Set only the first channel to 1 (green lines)
                img_color(seq(y-draw_len, y+draw_len), x, 0) = 0.f;
                img_color(seq(y-draw_len, y+draw_len), x, 1) = 1.f;
                img_color(seq(y-draw_len, y+draw_len), x, 2) = 0.f;
                good_corners++;
            }
        }
    }

    // Previews color image with green crosshairs
    std::cout << "Corners detected: " << good_corners << std::endl;
    std::cout << "Saving Image..." << std::endl;
    saveimage("test.png", img_color);
    return 0;
}
