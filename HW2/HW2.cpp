/********************************************************
* Filename    : HW2.cpp
* Author      : Stanley Chueh
* Note        : ADIP HW2 - menu-driven (portable)
*********************************************************/

#define _CRT_SECURE_NO_DEPRECATE

#include <cstdio>    // fopen, fread, fwrite, fclose, perror
#include <cstdint>   // uint8_t
#include <cstring>   // memcpy
#include <cstdlib>   // rand
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Shinking the image lena512.raw with ratio 1:2 using row-column deletion.
static void task_1_1(const char* input_image, const char* output_image_blur, const char* output_image_no_blur) 
{
    const int W = 512, H = 512;
    FILE* input_file = std::fopen(input_image, "rb");

    // Read the input image into a vector
    std::vector<unsigned char> image_original(W * H);
    std::fread(image_original.data(), 1, W * H, input_file);
    std::fclose(input_file);

    // vector for blurred image
    std::vector<unsigned char> image_blur(W * H);

    // 3x3 average filter
    for (int r = 1; r < H - 1; ++r) {
        for (int c = 1; c < W - 1; ++c) {
            int sum = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    sum += image_original[(r + dr) * W + (c + dc)];
                }
            }
            image_blur[r * W + c] = sum / 9;
        }
    }

    // Create a vector for the shrunk image
    std::vector<unsigned char> image_shrink((W / 2) * (H / 2));
    std::vector<unsigned char> image_shrink_blur((W / 2) * (H / 2));

    // row-column deletion
    for (int r = 0; r < H; r += 2)
    {
        for (int c = 0; c < W; c += 2)
        {
            image_shrink[(r / 2) * (W / 2) + (c / 2)] = image_original[r * W + c];
            image_shrink_blur[(r / 2) * (W / 2) + (c / 2)] = image_blur[r * W + c];
        }
    }

    // Write the shrunk image to the output file
    FILE* output_file = std::fopen(output_image_blur, "wb");
    FILE* output_file_no_blur = std::fopen(output_image_no_blur, "wb");

    std::fwrite(image_shrink_blur.data(), 1, (W / 2) * (H / 2), output_file);
    std::fwrite(image_shrink.data(), 1, (W / 2) * (H / 2), output_file_no_blur);

    std::fclose(output_file_no_blur);
    std::fclose(output_file);
}

// Use nearest neighbor interpolation and bilinear algorithm to reaize the image lena256.raw into size 384x384.
static void task_1_2(const char* input_image)
{
    const int W = 256, H = 256;
    FILE* input_file = std::fopen(input_image, "rb");
    if (!input_file) {
        perror("Error opening input file");
        return;
    }

    std::vector<unsigned char> image_original(W * H);
    std::fread(image_original.data(), 1, W * H, input_file);
    std::fclose(input_file);

    const int new_W = 384, new_H = 384; // 1.5 times
    const int src_W = 256, src_H = 256;

    // Nearest Neighbor Interpolation
    std::vector<unsigned char> image_nn(new_W * new_H);
    for (int r = 0; r < new_H; r++) {
        for (int c = 0; c < new_W; c++) {
            int src_r = r * H / new_H;
            int src_c = c * W / new_W;
            image_nn[r * new_W + c] = image_original[src_r * W + src_c];
        }
    }

    FILE* output_file_nn = std::fopen("task1_2_nearest_neighbor.raw", "wb");
    if (!output_file_nn) {
        perror("Error opening output file for Nearest Neighbor");
        return;
    }
    std::fwrite(image_nn.data(), 1, new_W * new_H, output_file_nn);
    std::fclose(output_file_nn);

    // Bilinear Interpolation
    std::vector<unsigned char> image_bilinear(new_W * new_H);
    for (int r = 0; r < new_H; r++) {
        for (int c = 0; c < new_W; c++) {
            // Map the pixel (r, c) in the new image to the corresponding position in the original image
            float src_r = (float)r * (H - 1) / (new_H - 1);
            float src_c = (float)c * (W - 1) / (new_W - 1);
            // Find the coordinates of the 4 neighboring pixels
            int r1 = (int)src_r;
            int c1 = (int)src_c;
            int r2 = std::min(r1 + 1, H - 1);
            int c2 = std::min(c1 + 1, W - 1);

            // Compute the weights for interpolation
            float dr = src_r - r1;
            float dc = src_c - c1;

            // Bilinear interpolation formula
            float val =
                (1 - dr) * (1 - dc) * image_original[r1 * W + c1] +
                (1 - dr) * dc * image_original[r1 * W + c2] +
                dr * (1 - dc) * image_original[r2 * W + c1] +
                dr * dc * image_original[r2 * W + c2];

            image_bilinear[r * new_W + c] = static_cast<unsigned char>(val + 0.5f);
        }
    }

    FILE* output_file_bilinear = std::fopen("task1_2_bilinear.raw", "wb");
    if (!output_file_bilinear) {
        perror("Error opening output file for Bilinear");
        return;
    }
    std::fwrite(image_bilinear.data(), 1, new_W * new_H, output_file_bilinear);
    std::fclose(output_file_bilinear);
}

// Grouping pixels based on their intensity values and spatial connectivity.
struct Pixel {
    int r, c; // row, column
};

static int find_interval(unsigned char val)
{
    if (val <= 57) return 0;
    else if (val <= 81) return 1;
    else if (val <= 105) return 2;
    else if (val <= 129) return 3;
    else if (val <= 153) return 4;
    else if (val <= 177) return 5;
    else if (val <= 201) return 6;
    else return 7;
}

static void task2_1(const char* input_name, const char* output_name)
{
    const int W = 50, H = 50;   
    FILE* input_file = std::fopen(input_name, "rb");
    if (!input_file) {
        perror("Error opening input file");
        return;
    }

    // Read the input image into a vector
    std::vector<unsigned char> image_original(W * H);
    std::fread(image_original.data(), 1, W * H, input_file);
    std::fclose(input_file);

    // Output image and visited map
    std::vector<unsigned char> image_output(W * H);
    std::vector<int> visited(W * H);

    // 8-neighbor offsets
    const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    /*
        (-1,-1)  (-1,0)  (-1,1)
        (0,-1)   (0,0)   (0,1)
        (1,-1)   (1,0)   (1,1)
        
        row=-1,-1,-1....
        col=-1,0,1....
    */

    // BFS over all pixels
    for (int r = 0; r < H; ++r) 
    {
        for (int c = 0; c < W; ++c) {
            int idx = r * W + c;
            if (visited[idx]) 
                continue;

            int interval_id = find_interval(image_original[idx]); // find interval of current pixel

            // Start BFS for a connected component
            std::queue<Pixel> q;
            q.push({r, c}); 
            visited[idx] = 1; // mark as visited

            std::vector<int> component_pixels; 
            component_pixels.push_back(idx); // Similar to .append() in Python
            int sum = image_original[idx];

            while (q.size()) {
                Pixel p = q.front(); // get the front element(r,c)
                q.pop(); // remove the front element(FIFO)

                for (int k = 0; k < 8; ++k) // 8-neighbors
                { 
                    int nr = p.r + dr[k]; // dr=[-1,-1,-1, 0,0, 1,1,1]
                    int nc = p.c + dc[k]; // dc=[-1, 0, 1,-1,1,-1,0,1]
                    /*
                    (-1,-1)  (-1,0)  (-1,1)
                    (0,-1)   (0,0)   (0,1)
                    (1,-1)   (1,0)   (1,1)
                    */
                    if (nr < 0 || nr >= H || nc < 0 || nc >= W)  // boundary check
                        continue;
                    /*
                    Ex:
                    (0,0),(0,1),(0,2),(0,3)...
                    (1,0),(1,1),(1,2),(1,3)...
                    (2,0),(2,1),(2,2),(2,3)...
                    (3,0),(3,1),(3,2),(3,3)... 
                    if now in (0,0),and we want to check its 8-neighbors
                    (nr,nc) = (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1) => only (0,1),(1,0),(1,1) are valid
                    */
                    int nidx = nr * W + nc; // neighbor index(in-boundary)
                    if (visited[nidx]) // already visited
                        continue;

                    // must belong to same intensity interval
                    if (find_interval(image_original[nidx]) != interval_id) //check neighbor pixel's interval
                        continue;                                           // interval_id=>check current pixel's interval
                    // two conditions should be satisfied to be added into the queue
                    // two pixels are within 8-neighbors and belong to the same intensity interval

                    visited[nidx] = 1; // mark neighbor index as visited
                    q.push({nr, nc});  // add neighbor pixel into the queue
                    component_pixels.push_back(nidx); //add neighbor index into the component vector
                    sum += image_original[nidx]; // sum of all pixel values in the component
                }
            }

            // Compute mean intensity
            int mean_val = std::round(sum / (double)component_pixels.size()); 
            // the reason to add double is to make sure it will not discard the decimal part.
            // component_pixels.size() is the number of pixels in the component

            // Assign mean to all pixels in the component
            for (int i = 0; i<component_pixels.size();++i)
                image_output[component_pixels[i]] = static_cast<unsigned char>(mean_val);
            // ready for next component
        }
    }

    // Write output RAW file
    FILE* out_file = std::fopen(output_name, "wb");
    if (!out_file) {
        perror("Error opening output file");
        return;
    }
    std::fwrite(image_output.data(), 1, W * H, out_file);
    std::fclose(out_file);
}

// Compute MSE and PSNR between two images(lena_eye50.raw and lena_eye50_out.raw).
static void task2_2(const char* input_name, const char* output_name)
{
    const int W = 50, H = 50;  // image size
    FILE* f1 = std::fopen(input_name, "rb");
    FILE* f2 = std::fopen(output_name, "rb");

    if (!f1 || !f2) {
        perror("Error opening input files");
        return;
    }

    // Read both images
    std::vector<unsigned char> img1(W * H);
    std::vector<unsigned char> img2(W * H);
    std::fread(img1.data(), 1, W * H, f1);
    std::fread(img2.data(), 1, W * H, f2);
    std::fclose(f1);
    std::fclose(f2);

    // Compute MSE: https://en.wikipedia.org/wiki/Mean_squared_error
    double mse = 0.0;
    for (int i = 0; i < W * H; ++i) {
        double diff = static_cast<double>(img1[i]) - static_cast<double>(img2[i]);
        mse += diff * diff;
    }
    mse /= (W * H);
    // MSE is to compare the similarity between two images, the smaller the better.

    // Compute PSNR(Peak Signal-to-Noise Ratio): https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    // PSNR = 10 * log10((MAX_I^2) / MSE)
    double MAX_I = 255.0;
    double psnr = 10.0 * std::log10((MAX_I * MAX_I) / mse);
    // PSNR is also to compare the similarity between two images, the larger the better.

    // Output results
    std::cout << "Image Quality Metrics" << std::endl;
    std::cout << "MSE= " << mse << std::endl;
    std::cout << "PSNR= " << psnr << " dB" << std::endl;
}

int main() {
    std::cout << "----- Homework 1 Menu -----\n";
    while (true) {
        std::cout << "\n================ Results Menu ================\n"
                  << " 1) 1-1\n"
                  << " 2) 1-2\n"
                  << " 3) 2-1\n"
                  << " 4) 2-2\n"
                  << " 0) Exit\n"
                  << "Enter the question number: ";

        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(1 << 20, '\n');
            std::cout << "Invalid input. Please enter a number between 0 and 4.\n";
            continue;
        }
        if (choice == 0) break;

        switch (choice) {
            case 1: task_1_1("lena512.raw","task1_1_blur.raw","task1_1_no_blur.raw"); break;
            case 2: task_1_2("lena256.raw"); break;
            case 3: task2_1("lena_eye50.raw", "lena_eye50_out.raw"); break;
            case 4: task2_2("lena_eye50.raw", "lena_eye50_out.raw"); break;
            default: std::cout << "Unknown selection. Try 0-4.\n"; break;
        }
    }
    return 0;
}
