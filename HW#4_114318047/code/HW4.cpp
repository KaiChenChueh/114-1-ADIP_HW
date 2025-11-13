/********************************************************
* Filename    : HW4.cpp
* Author      : Stanley Chueh
* Note        : ADIP HW4 - menu-driven (portable)
*********************************************************/

#define _CRT_SECURE_NO_DEPRECATE

#include <cstdio>    // fopen, fread, fwrite, fclose, perror
#include <cstdint>   // uint8_t
#include <cstring>   // memcpy
#include <cstdlib>   // rand
#include <algorithm>  // sort, min, max
#include <cmath>      // sqrt, pow
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Helpers
std::vector<uint8_t> apply_replication_padding(const std::vector<uint8_t>& input, int width, int height, int padding) 
{
    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;
    std::vector<uint8_t> padded_input(padded_width * padded_height, 0);

    // Copy the original image into the padded image
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            padded_input[(r + padding) * padded_width + (c + padding)] = input[r * width + c];
        }
    }

    // Replicate borders
    for (int c = 0; c < width; ++c) {
        for (int p = 0; p < padding; ++p) {
            padded_input[p * padded_width + (c + padding)] = input[c]; // Top rows
            padded_input[(height + padding + p) * padded_width + (c + padding)] = input[(height - 1) * width + c]; // Bottom rows
        }
    }
    for (int r = 0; r < height; ++r) {
        for (int p = 0; p < padding; ++p) {
            padded_input[(r + padding) * padded_width + p] = input[r * width]; // Left columns
            padded_input[(r + padding) * padded_width + (width + padding + p)] = input[r * width + width - 1]; // Right columns
        }
    }

    // Corners
    for (int p = 0; p < padding; ++p) {
        for (int q = 0; q < padding; ++q) {
            padded_input[p * padded_width + q] = input[0]; // Top-left
            padded_input[p * padded_width + (width + padding + q)] = input[width - 1]; // Top-right
            padded_input[(height + padding + p) * padded_width + q] = input[(height - 1) * width]; // Bottom-left
            padded_input[(height + padding + p) * padded_width + (width + padding + q)] = input[(height - 1) * width + width - 1]; // Bottom-right
        }
    }

    return padded_input;
}

std::vector<uint8_t> apply_gaussian_filter(const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int width, int height, int kernel_size) 
{
    const float kernel_3x3[3][3] = {
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
    };

    const float kernel_5x5[5][5] = {
        {1 / 273.0f, 4 / 273.0f, 6 / 273.0f, 4 / 273.0f, 1 / 273.0f},
        {4 / 273.0f,16 / 273.0f,24 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {6 / 273.0f,24 / 273.0f,36 / 273.0f,24 / 273.0f, 6 / 273.0f},
        {4 / 273.0f,16 / 273.0f,24 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {1 / 273.0f, 4 / 273.0f, 6 / 273.0f, 4 / 273.0f, 1 / 273.0f}
    };

    const float kernel_7x7[7][7] = {
        {0.000036, 0.000363, 0.001446, 0.002290, 0.001446, 0.000363, 0.000036},
        {0.000363, 0.003676, 0.014662, 0.023223, 0.014662, 0.003676, 0.000363},
        {0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446},
        {0.002290, 0.023223, 0.092651, 0.146768, 0.092651, 0.023223, 0.002290},
        {0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446},
        {0.000363, 0.003676, 0.014662, 0.023223, 0.014662, 0.003676, 0.000363},
        {0.000036, 0.000363, 0.001446, 0.002290, 0.001446, 0.000363, 0.000036}
    };

    const float* kernel = nullptr;
    if (kernel_size == 3) { 
        kernel = &kernel_3x3[0][0];
    } else if (kernel_size == 5) { 
        kernel = &kernel_5x5[0][0];
    } else if (kernel_size == 7) {
        kernel = &kernel_7x7[0][0];
    } else {
        std::cerr << "Unsupported kernel size. Only 3x3, 5x5, and 7x7 are supported." << std::endl;
        return output;
    }

    int padding = kernel_size / 2; 
    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;

    // Create a padded image with replicated borders(replication padding)
    std::vector<uint8_t> padded_input((padded_width) * (padded_height));
    /*
        for 3x3, padding = 1
        for 5x5, padding = 2

        // Padding logic
        original size of image:width x height
        padded size of image:
        padded_height = height + 2*padding
        padded_width  = width  + 2*padding

        That is, for point in original image (r,c), it maps to (r+padding, c+padding) in padded image

        recall:
        for width x height image:
        index = r * width + c

        for padded image:
        r' = r + padding
        c' = c + padding
        index' = r' * (width + 2*padding) + c' = r' * padded_width + c'

    */
    padded_input = apply_replication_padding(input, width, height, padding);

    // Apply Gaussian filter
    /*
    starts from (r,c) = (padding,padding) cause after padding,
    the origin (0,0) maps to (padding,padding)
    */
    for (int r = padding; r < height + padding; ++r) {
        for (int c = padding; c < width + padding; ++c) {
            float sum = 0.0f;
            /*
            (r-1,c-1) (r-1,c) (r-1,c+1)
            (r,c-1)   (r,c)   (r,c+1)
            (r+1,c-1) (r+1,c) (r+1,c+1)
            */
            for (int kr = -padding; kr <= padding; ++kr) {
                for (int kc = -padding; kc <= padding; ++kc) {
                    // (r+kr) * padded_width + (c+kc) = index in padded image
                    // (kr + padding) * kernel_size + (kc + padding) = index in kernel
                    sum += padded_input[(r + kr) * (padded_width) + (c + kc)] * kernel[(kr + padding) * kernel_size + (kc + padding)];
                }
            }
            // sum was defined as float, need to cast to uint8_t
            output[(r - padding) * width + (c - padding)] = static_cast<uint8_t>(sum);
        }
    }

    return output;
}
// median filter
std::vector<uint8_t> apply_median_filter(const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int width, int height, int kernel_size) 
{
    int padding = kernel_size / 2;
    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;

    // Create a padded image with replicated borders
    std::vector<uint8_t> padded_input((padded_width) * (padded_height));
    padded_input = apply_replication_padding(input, width, height, padding);

    // Apply Median filter
    for (int r = padding; r < height + padding; ++r) {
        for (int c = padding; c < width + padding; ++c) {
            // 3x3 or 5x5 neighborhood
            std::vector<uint8_t> neighborhood;
            for (int kr = -padding; kr <= padding; ++kr) {
                for (int kc = -padding; kc <= padding; ++kc) {
                    // append pixel in padded_input to neighborhood
                    neighborhood.push_back(padded_input[(r + kr) * (padded_width) + (c + kc)]);
                }
            }
            // sort and find median
            std::sort(neighborhood.begin(), neighborhood.end());
            output[(r - padding) * width + (c - padding)] = neighborhood[neighborhood.size() / 2];
        }
    }
    return output;
}
// sobel filter
std::vector<uint8_t> apply_sobel_filter(const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int width, int height, bool horizontal) 
{
    // sobel kernels
    const int8_t sobel_x[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    const int8_t sobel_y[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    const int8_t* kernel;
    if (horizontal) {
        kernel = &sobel_x[0][0];
    } else {
        kernel = &sobel_y[0][0];
    }

    int padding = 1;
    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;

    // Create a padded image with replicated borders
    std::vector<uint8_t> padded_input((padded_width) * (padded_height));
    padded_input = apply_replication_padding(input, width, height, padding);

    // Apply Sobel filter
    for (int r = padding; r < height + padding; ++r) {
        for (int c = padding; c < width + padding; ++c) {
            int sum = 0;
            for (int kr = -padding; kr <= padding; ++kr) {
                for (int kc = -padding; kc <= padding; ++kc) {
                    sum += padded_input[(r + kr) * (padded_width) + (c + kc)] * kernel[(kr + padding) * 3 + (kc + padding)];
                }
            }
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            
            output[(r - padding) * width + (c - padding)] = sum;
        }
    }

    return output;
}

// Apply laplacian filter
std::vector<uint8_t> apply_laplacian_filter(const char* input_file, const char* output, int n)
{
    // apply 3x3 Laplacian filter using 4-neighbor and 8-neighbor
    const int8_t laplacian_4_neighbor[3][3] = {
        { 0, -1,  0},
        {-1,  4, -1},
        { 0, -1,  0}
    };

    const int8_t laplacian_8_neighbor[3][3] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    // const int8_t (*kernel)[3] = (n == 4) ? laplacian_4_neighbor : laplacian_8_neighbor;
    const int8_t* kernel;
    if (n == 4) {
        kernel = &laplacian_4_neighbor[0][0];
    } else {
        kernel = &laplacian_8_neighbor[0][0];
    }

    int width = 600, height = 400;
    int padding = 1;
    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;

    std::vector<uint8_t> input_image(width * height);
    std::vector<uint8_t> output_image(width * height, 0);

    FILE* input = fopen(input_file, "rb");
    if (!input) {
        perror("Failed to open input file");
        return output_image;
    }
    std::fread(input_image.data(), 1, input_image.size(), input);
    std::fclose(input);

    std::vector<uint8_t> padded_input((padded_width) * (padded_height));

    // Padding replication
    padded_input = apply_replication_padding(input_image, width, height, padding);

    // Apply Laplacian filter
    for (int r = padding; r < height + padding; ++r) {
        for (int c = padding; c < width + padding; ++c) {
            int sum = 0;
            for (int kr = -padding; kr <= padding; ++kr) {
                for (int kc = -padding; kc <= padding; ++kc) {
                    sum += padded_input[(r + kr) * (padded_width) + (c + kc)] * kernel[(kr + padding) * 3 + (kc + padding)];
                }
            }
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output_image[(r - padding) * width + (c - padding)] = sum;
        }
    }

    return output_image;
}

std::vector<uint8_t> apply_high_boost_filter(const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int width, int height, int kernel_size, float A) 
{
    // High-boost filtering: output = A * original - blurred
    std::vector<uint8_t> blurred_image(width * height);
    blurred_image = apply_gaussian_filter(input, blurred_image, width, height, kernel_size);

    for (int i = 0; i < width * height; ++i) {
        int high_boost_value = static_cast<int>(A * static_cast<int>(input[i]) - static_cast<int>(blurred_image[i]));
        if (high_boost_value > 255)
            output[i] = 255;
        else if (high_boost_value < 0)
            output[i] = 0;
        else
            output[i] = high_boost_value;
    }

    return output;
}

// Egde detection 
static void task_1_1_i(const char* input_file,const char* output_file_gaussian3x3,const char* output_file_gaussian5x5)
{
    // Apply gaussian smoothing filters with kernel sizes of 3x3 and 5x5 to reduce salt-and-pepper noise
    FILE* input = fopen(input_file, "rb");
    if (!input) {
        perror("Failed to open input file");
        return;
    }

    const int width = 600;
    const int height = 400;

    std::vector<uint8_t> image_data(width * height);
    std::fread(image_data.data(), 1, image_data.size(), input);
    std::fclose(input);

    // Apply 3x3 Gaussian filter
    std::vector<uint8_t> output_gaussian3x3(width * height);
    std::vector<uint8_t> output_gaussian5x5(width * height);

    output_gaussian3x3 = apply_gaussian_filter(image_data, output_gaussian3x3, width, height,3);
    output_gaussian5x5 = apply_gaussian_filter(image_data, output_gaussian5x5, width, height,5);

    // Save output images
    FILE* output3x3 = fopen(output_file_gaussian3x3, "wb");
    FILE* output5x5 = fopen(output_file_gaussian5x5, "wb");
    if (!output3x3 || !output5x5) {
        perror("Failed to open output file for 3x3 or 5x5 Gaussian");
        return;
    }
    std::fwrite(output_gaussian3x3.data(), 1, output_gaussian3x3.size(), output3x3);
    std::fwrite(output_gaussian5x5.data(), 1, output_gaussian5x5.size(), output5x5);
    std::fclose(output3x3);
    std::fclose(output5x5);

    // save as png for easy viewing    
    cv::Mat img3x3(height, width, CV_8UC1, output_gaussian3x3.data());
    cv::Mat img5x5(height, width, CV_8UC1, output_gaussian5x5.data());
    cv::Mat img_original(height, width, CV_8UC1, image_data.data());

    cv::imwrite("parthenon_gaussian3x3.png", img3x3);
    cv::imwrite("parthenon_gaussian5x5.png", img5x5);
    cv::imwrite("parthenon_original.png", img_original);

}

static void task_1_1_ii(const char* input_file,const char* output_file_gaussian3x3,const char* output_file_gaussian5x5)
{
    // apply median filters with kernel sizes of 3x3 and 5x5 to reduce salt-and-pepper noise
    FILE* input = fopen(input_file, "rb");
    if (!input) {
        perror("Failed to open input file");
        return;
    }
    const int width = 600;
    const int height = 400;
    std::vector<uint8_t> image_data(width * height);
    std::fread(image_data.data(), 1, image_data.size(), input);
    std::fclose(input);

    // Apply 3x3 Median filter
    std::vector<uint8_t> output_median3x3(width * height);
    std::vector<uint8_t> output_median5x5(width * height);

    output_median3x3 = apply_median_filter(image_data, output_median3x3, width, height,3);
    output_median5x5 = apply_median_filter(image_data, output_median5x5, width, height,5);

    // Save output images
    FILE* output_median3x3_file = fopen("parthenon_median3x3.raw", "wb");
    FILE* output_median5x5_file = fopen("parthenon_median5x5.raw", "wb");

    if (!output_median3x3_file || !output_median5x5_file) {
        perror("Failed to open output file for 3x3 or 5x5 Median");
        return;
    }

    std::fwrite(output_median3x3.data(), 1, output_median3x3.size(), output_median3x3_file);
    std::fwrite(output_median5x5.data(), 1, output_median5x5.size(), output_median5x5_file);

    std::fclose(output_median3x3_file);
    std::fclose(output_median5x5_file);

    // save as png for easy viewing    
    cv::Mat img_median3x3(height, width, CV_8UC1, output_median3x3.data());
    cv::Mat img_median5x5(height, width, CV_8UC1, output_median5x5.data());
    cv::Mat img_original(height, width, CV_8UC1, image_data.data());

    cv::imwrite("parthenon_median3x3.png", img_median3x3);
    cv::imwrite("parthenon_median5x5.png", img_median5x5);
    cv::imwrite("parthenon_original.png", img_original);

}

static void task_1_1_iii()
{
    std::cout<<"Please view report\n";
}

static void task_1_2_a(const char* input_file,const char* output_file_mask,const char* output_file_sharpen)
{
    // unsharp masking(parthenon_median3x3)
    /*
    Implement a Gaussian 3x3 filter to blur image,and then use unsharp masking to sharpen the original image.
    */
    FILE* input = fopen(input_file, "rb");
    if (!input) {
        perror("Failed to open input file");
        return;
    }

    const int width = 600;
    const int height = 400;
    std::vector<uint8_t> image_data(width * height);
    std::fread(image_data.data(), 1, image_data.size(), input);
    std::fclose(input);

    // Create blurred image using Gaussian filter
    std::vector<uint8_t> mask_image(width * height);
    std::vector<uint8_t> blurred_image(width * height);
    blurred_image = apply_gaussian_filter(image_data, blurred_image, width, height,3);

    // Unsharp masking: image + mask_image
    std::vector<uint8_t> unsharp_image(width * height);

    for (int i = 0; i < width * height; ++i) {
        int mask_value = static_cast<int>(image_data[i]) - static_cast<int>(blurred_image[i]);
        int sharpened_value = static_cast<int>(image_data[i]) + (mask_value);
        if (sharpened_value > 255)
            unsharp_image[i] = 255;
        else if (sharpened_value < 0)
            unsharp_image[i] = 0;
        else
            unsharp_image[i] = sharpened_value;

        // shift to middle-gray (0 difference = gray)
        int mask_value_clear = mask_value + 128;   
        if (mask_value_clear  < 0) mask_value_clear  = 0;
        if (mask_value_clear  > 255) mask_value_clear  = 255;
        mask_image[i] = mask_value_clear ;

    }

    // Save output images
    FILE* output_mask = fopen(output_file_mask, "wb");
    FILE* output_sharpen = fopen(output_file_sharpen, "wb");
    if (!output_mask || !output_sharpen) {
        perror("Failed to open output file for unsharp masking or sharpen");
        return;
    }

    std::fwrite(mask_image.data(), 1, mask_image.size(), output_mask);
    std::fwrite(unsharp_image.data(), 1, unsharp_image.size(), output_sharpen);
    std::fclose(output_mask);
    std::fclose(output_sharpen);

    // save as png for easy viewing  
    cv::Mat img_mask(height, width, CV_8UC1, mask_image.data());
    cv::Mat img_unsharp(height, width, CV_8UC1, unsharp_image.data());
    cv::imwrite("unsharp_mask.png", img_mask);
    cv::imwrite("parthenon_unsharp_mask.png", img_unsharp);
}

static void task_1_2_b_i(const char* input_file,const char* output_file_dege,const char* output_file_edge)
{
    // compute the horizontal and vertical edge maps of the image using the Sobel operator
    FILE* input = fopen(input_file, "rb");
    if (!input) {
        perror("Failed to open input file");
        return;
    }
    const int width = 600;
    const int height = 400;

    std::vector<uint8_t> image_data(width * height);
    std::fread(image_data.data(), 1, image_data.size(), input);
    std::fclose(input);

    // Apply Sobel filter
    std::vector<uint8_t> output_sobel_horizontal(width * height);
    std::vector<uint8_t> output_sobel_vertical(width * height);

    output_sobel_horizontal = apply_sobel_filter(image_data, output_sobel_horizontal, width, height, true);
    output_sobel_vertical = apply_sobel_filter(image_data, output_sobel_vertical, width, height, false);

    // Save output images
    FILE* output_dege = fopen(output_file_dege, "wb");
    FILE* output_edge = fopen(output_file_edge, "wb");
    if (!output_dege || !output_edge) {
        perror("Failed to open output file for Sobel horizontal or vertical");
        return;
    }

    std::fwrite(output_sobel_horizontal.data(), 1, output_sobel_horizontal.size(), output_dege);
    std::fwrite(output_sobel_vertical.data(), 1, output_sobel_vertical.size(), output_edge);

    std::fclose(output_dege);
    std::fclose(output_edge);

    // save as png for easy viewing
    cv::Mat img_sobel_horizontal(height, width, CV_8UC1, output_sobel_horizontal.data());
    cv::Mat img_sobel_vertical(height, width, CV_8UC1, output_sobel_vertical.data());
    cv::imwrite("sobel_0_dege.png", img_sobel_horizontal);
    cv::imwrite("sobel_90_edge.png", img_sobel_vertical);
}

static void task_1_2_b_ii(const char* input_file_0_dege,const char* input_file_90_edge,const char* output_file)
{
    // combine the two gradient images to obtain sobel_combined_edge.raw
    FILE* input_0_dege = fopen(input_file_0_dege, "rb");
    FILE* input_90_edge = fopen(input_file_90_edge, "rb");

    if (!input_0_dege || !input_90_edge) {
        perror("Failed to open input file");
        return;
    }
    const int width = 600;
    const int height = 400;

    std::vector<uint8_t> image_data_0_dege(width * height);
    std::vector<uint8_t> image_data_90_edge(width * height);

    std::fread(image_data_0_dege.data(), 1, image_data_0_dege.size(), input_0_dege);
    std::fread(image_data_90_edge.data(), 1, image_data_90_edge.size(), input_90_edge);

    std::fclose(input_0_dege);
    std::fclose(input_90_edge);

    // Combine gradient images
    /*
    Magnitude = sqrt(Gx^2 + Gy^2)
    */
    std::vector<uint8_t> combined_image(width * height);
    for (int i = 0; i < width * height; ++i) {
        int combined_value = static_cast<int>(std::sqrt(std::pow(image_data_0_dege[i], 2) + std::pow(image_data_90_edge[i], 2)));
        if (combined_value > 255)
            combined_image[i] = 255;
        else if(combined_value < 0)
            combined_image[i] = 0;
        else
            combined_image[i] = combined_value;
    }

    // Save output image
    FILE* output_combined = fopen(output_file, "wb");
    if (!output_combined) {
        perror("Failed to open output file for combined edge");
        return;
    }
    std::fwrite(combined_image.data(), 1, combined_image.size(), output_combined);
    std::fclose(output_combined);

    // save as png for easy viewing
    cv::Mat img_combined(height, width, CV_8UC1, combined_image.data());
    cv::imwrite("sobel_combined_edge.png", img_combined);
}

static void task_1_2_b_iii(const char* input_file_median3x3,const char* input_file_sobel_combined,const char* output_file)
{
    // sharpen the original image using the combined edge map obtained in step 2-2-2
    FILE* input_median3x3 = fopen(input_file_median3x3, "rb");
    FILE* input_sobel_combined = fopen(input_file_sobel_combined, "rb");
    if (!input_median3x3 || !input_sobel_combined) {
        perror("Failed to open input file");
        return;
    }
    const int width = 600;
    const int height = 400;
    std::vector<uint8_t> image_data_median3x3(width * height);
    std::vector<uint8_t> image_data_sobel_combined(width * height);

    std::fread(image_data_median3x3.data(), 1, image_data_median3x3.size(), input_median3x3);
    std::fread(image_data_sobel_combined.data(), 1, image_data_sobel_combined.size(), input_sobel_combined);

    std::fclose(input_median3x3);
    std::fclose(input_sobel_combined);

    // Sharpen image using combined edge map
    std::vector<uint8_t> sharpened_image(width * height);
    for (int i = 0; i < width * height; ++i) {
        //int sharpened_value = static_cast<int>(image_data_median3x3[i]) + static_cast<int>(image_data_sobel_combined[i]);
        // imitate unsharp masking
        int mask = -0.35 * static_cast<int>(image_data_sobel_combined[i]);
        int sharpened_value = static_cast<int>(image_data_median3x3[i]) +  mask;
        sharpened_image[i] = static_cast<uint8_t>(std::min(std::max(sharpened_value, 0), 255));
    }
    // Save output image
    FILE* output_sharpened = fopen(output_file, "wb");

    if (!output_sharpened) {
        perror("Failed to open output file for sharpened image");
        return;
    }
    std::fwrite(sharpened_image.data(), 1, sharpened_image.size(), output_sharpened);

    std::fclose(output_sharpened);

    // save as png for easy viewing
    cv::Mat img_sharpened(height, width, CV_8UC1, sharpened_image.data());
    cv::imwrite("parthenon_sobel_sharpened.png", img_sharpened);
}

static void task_1_2_c_i(const char* input_file, const char* output_file_4_neighbor, const char* output_file_8_neighbor)
{
    FILE* input_median3x3 = fopen(input_file, "rb");

    if (!input_median3x3) {
        perror("Failed to open input file");
        return;
    }
    const int width = 600;
    const int height = 400;

    std::vector<uint8_t> image_data_median3x3(width * height);
    std::fread(image_data_median3x3.data(), 1, image_data_median3x3.size(), input_median3x3);
    std::fclose(input_median3x3);

    // Apply Laplacian filter
    std::vector<uint8_t> laplacian_image_4 = apply_laplacian_filter(input_file, output_file_4_neighbor, 4);
    std::vector<uint8_t> laplacian_image_8 = apply_laplacian_filter(input_file, output_file_8_neighbor, 8);
    
    // Save output image
    FILE* output_laplacian_4 = fopen(output_file_4_neighbor, "wb");
    FILE* output_laplacian_8 = fopen(output_file_8_neighbor, "wb");
    if (!output_laplacian_4 || !output_laplacian_8) {
        perror("Failed to open output file for Laplacian image");
        return;
    }

    std::fwrite(laplacian_image_4.data(), 1, laplacian_image_4.size(), output_laplacian_4);
    std::fwrite(laplacian_image_8.data(), 1, laplacian_image_8.size(), output_laplacian_8);

    std::fclose(output_laplacian_4);
    std::fclose(output_laplacian_8);

    // save as png for easy viewing
    cv::Mat img_laplacian_4(height, width, CV_8UC1, laplacian_image_4.data());
    cv::Mat img_laplacian_8(height, width, CV_8UC1, laplacian_image_8.data());

    cv::imwrite("laplacian_4neighbor_edge.png", img_laplacian_4);
    cv::imwrite("laplacian_8neighbor_edge.png", img_laplacian_8);
}

static void task_1_2_c_ii(const char* input_file_median3x3, const char* input_file_laplacian_4,const char* input_file_laplacian_8,const char* output_file_sharpened_4neighbor,const char* output_file_sharpened_8neighbor)
{
    // sharpen the original image using both Laplacian edge
    FILE* input_file = fopen(input_file_median3x3, "rb");
    FILE* input_laplacian_4 = fopen(input_file_laplacian_4, "rb");
    FILE* input_laplacian_8 = fopen(input_file_laplacian_8, "rb");
    if (!input_file || !input_laplacian_4 || !input_laplacian_8) {
        perror("Failed to open input file");
        return;
    }

    const int width = 600;
    const int height = 400;

    std::vector<uint8_t> image_data(width * height);
    std::vector<uint8_t> image_data_laplacian_4(width * height);
    std::vector<uint8_t> image_data_laplacian_8(width * height);

    std::fread(image_data.data(), 1, image_data.size(), input_file);
    std::fread(image_data_laplacian_4.data(), 1, image_data_laplacian_4.size(), input_laplacian_4);
    std::fread(image_data_laplacian_8.data(), 1, image_data_laplacian_8.size(), input_laplacian_8);

    std::fclose(input_file);
    std::fclose(input_laplacian_4);
    std::fclose(input_laplacian_8);

    // Sharpen image using laplacian result
    std::vector<uint8_t> sharpened_image_4(width * height);
    std::vector<uint8_t> sharpened_image_8(width * height);

    for (int i = 0; i < width * height; ++i) {
        int sharpened_value_4 = static_cast<int>(image_data[i] + static_cast<int>(image_data_laplacian_4[i]));
        int sharpened_value_8 = static_cast<int>(image_data[i] + static_cast<int>(image_data_laplacian_8[i]));
        if (sharpened_value_4 > 255) 
            sharpened_image_4[i] = 255;
        else if (sharpened_value_4 < 0)
            sharpened_image_4[i] = 0;
        else
            sharpened_image_4[i] = static_cast<uint8_t>(std::min(std::max(sharpened_value_4, 0), 255));
        if (sharpened_value_8 > 255)
            sharpened_image_8[i] = 255;
        else if (sharpened_value_8 < 0)
            sharpened_image_8[i] = 0;
        else
            sharpened_image_8[i] = static_cast<uint8_t>(std::min(std::max(sharpened_value_8, 0), 255));
    }

    // Save output image
    FILE* output_sharpened_4 = fopen(output_file_sharpened_4neighbor, "wb");
    FILE* output_sharpened_8 = fopen(output_file_sharpened_8neighbor, "wb");
    if (!output_sharpened_4 || !output_sharpened_8) {
        perror("Failed to open output file for sharpened image");
        return;
    }
    std::fwrite(sharpened_image_4.data(), 1, sharpened_image_4.size(), output_sharpened_4);
    std::fwrite(sharpened_image_8.data(), 1, sharpened_image_8.size(), output_sharpened_8);

    std::fclose(output_sharpened_4);
    std::fclose(output_sharpened_8);

    // save as png for easy viewing
    cv::Mat img_sharpened_4(height, width, CV_8UC1, sharpened_image_4.data());
    cv::Mat img_sharpened_8(height, width, CV_8UC1, sharpened_image_8.data());

    cv::imwrite("parthenon_laplacian_4neighbor_sharpened.png", img_sharpened_4);
    cv::imwrite("parthenon_laplacian_8neighbor_sharpened.png", img_sharpened_8);
}

static void task_1_2_c_iii()
{
    std::cout<<"Please view report\n";
}

static void task_2_1(const char* input,const char* output_ROI)
{   
    // find the ROI of flower_512x384.raw and save it as ROI.raw
    // extract the flowers as a ROI though a combination of blurring and thresholding
    // obtain a binary mask image where the flowers appear as white foreground regions and the leaves as black background.
    FILE* input_file = fopen(input, "rb");
    if (!input_file) {
        perror("Failed to open input file");
        return;
    }
    const int width = 512;
    const int height = 384;

    std::vector<uint8_t> image_data(width * height);
    std::fread(image_data.data(), 1, image_data.size(), input_file);
    std::fclose(input_file);

    // Apply Gaussian filter to blur the image(INCREASE TO 7x7 KERNEL TO BETTER BLURRING)
    std::vector<uint8_t> blurred_image(width * height);
    blurred_image = apply_gaussian_filter(image_data, blurred_image, width, height,7);

    // Thresholding to create binary mask
    std::vector<uint8_t> binary_mask(width * height);
    const uint8_t threshold = 200; // Adjust threshold value as needed
    for (int i = 0; i < width * height; ++i) {
        if (blurred_image[i] > threshold) {
            binary_mask[i] = 255; // White for foreground (flowers)
        } else {
            binary_mask[i] = 0;   // Black for background (leaves)
        }
    }
    // Save output binary mask
    FILE* output_file = fopen(output_ROI, "wb");
    if (!output_file) {
        perror("Failed to open output file for ROI");
        return;
    }
    std::fwrite(binary_mask.data(), 1, binary_mask.size(), output_file);
    std::fclose(output_file);

    // save as png for easy viewing
    cv::Mat img_binary_mask(height, width, CV_8UC1, binary_mask.data());
    cv::imwrite("ROI.png", img_binary_mask);
}

static void task_2_2(const char* input_ROI, const char* output_enhanced)
{
    // Using the binary ROI image from the previous step, perform different filtering operations on the
    // foreground and background. Specifically, apply a 3x3 high-boost filter to enhance the foreground(flowers),
    // and a 7x7 Gaussian filter to soften the background(leaves).
    // Use logical operations to combine these two results based on the ROI mask and produce the final output image.
    FILE* input_ROI_file = fopen(input_ROI, "rb");
    if (!input_ROI_file) {
        perror("Failed to open input ROI file");
        return;
    }
    const int width = 512;
    const int height = 384;
    std::vector<uint8_t> roi_data(width * height);
    std::fread(roi_data.data(), 1, roi_data.size(), input_ROI_file);
    std::fclose(input_ROI_file);

    // Load original flower image
    FILE* input_flower_file = fopen("flower_512x384.raw", "rb");
    if (!input_flower_file) {
        perror("Failed to open input flower file");
        return;
    }

    std::vector<uint8_t> flower_data(width * height);
    std::fread(flower_data.data(), 1, flower_data.size(), input_flower_file);
    std::fclose(input_flower_file);

    // Apply high-boost filter to foreground
    std::vector<uint8_t> high_boosted_image(width * height);
    high_boosted_image = apply_high_boost_filter(flower_data, high_boosted_image, width, height, 3, 1.7); // k=1.5
    
    // Apply Gaussian filter to background
    std::vector<uint8_t> blurred_image(width * height);
    blurred_image = apply_gaussian_filter(flower_data, blurred_image, width, height,7);

    // Combine based on ROI mask
    std::vector<uint8_t> final_image(width * height);
    for (int i = 0; i < width * height; ++i) {
        if (roi_data[i] == 255) { // Foreground
            final_image[i] = high_boosted_image[i];
        } else { // Background
            final_image[i] = blurred_image[i];
        }
    }

    // Save output enhanced image
    FILE* output_file = fopen(output_enhanced, "wb");
    if (!output_file) {
        perror("Failed to open output file for enhanced image");
        return;
    }
    std::fwrite(final_image.data(), 1, final_image.size(), output_file);
    std::fclose(output_file);

    // save as png for easy viewing
    cv::Mat img_final(height, width, CV_8UC1, final_image.data());
    cv::imwrite("flower_enhanced.png", img_final);
}

int main() {
    std::cout << "----- Homework 4 Menu -----\n";
    while (true) {
        std::cout << "\n================ Results Menu ================\n"
                  << " 1) task_1_1_i\n" 
                  << " 2) task_1_1_ii\n"
                  << " 3) task_1_1_iii\n"
                  << " 4) task_1_2_a\n"
                  << " 5) task_1_2_b_i\n"
                  << " 6) task_1_2_b_ii\n"
                  << " 7) task_1_2_b_iii\n"
                  << " 8) task_1_2_c_i\n"
                  << " 9) task_1_2_c_ii\n"
                  << " 10)task_1_2_c_iii\n"
                  << " 11)task_2_1\n"
                  << " 12)task_2_2\n"
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
            case 1: task_1_1_i("parthenon_600x400.raw","parthenon_gaussian3x3.raw","parthenon_gaussian5x5.raw"); break;
            case 2: task_1_1_ii("parthenon_600x400.raw","parthenon_median3x3.raw","parthenon_median5x5.raw"); break;
            case 3: task_1_1_iii(); break;
            case 4: task_1_2_a("parthenon_median3x3.raw","unsharp_mask.raw","parthenon_unsharp_mask.raw"); break;
            case 5: task_1_2_b_i("parthenon_median3x3.raw","sobel_0_dege.raw","sobel_90_edge.raw"); break;
            case 6: task_1_2_b_ii("sobel_0_dege.raw","sobel_90_edge.raw","sobel_combined_edge.raw"); break;
            case 7: task_1_2_b_iii("parthenon_median3x3.raw","sobel_combined_edge.raw","parthenon_sobel_sharpened.raw"); break;
            case 8: task_1_2_c_i("parthenon_median3x3.raw","laplacian_4neighbor_edge.raw","laplacian_8neighbor_edge.raw"); break;
            case 9: task_1_2_c_ii("parthenon_median3x3.raw","laplacian_4neighbor_edge.raw","laplacian_8neighbor_edge.raw","parthenon_laplacian_4neighbor_sharpened.raw","parthenon_laplacian_8neighbor_sharpened.raw"); break;
            case 10:task_1_2_c_iii(); break;
            case 11:task_2_1("flower_512x384.raw","ROI.raw"); break;
            case 12:task_2_2("ROI.raw","flower_enhanced.raw"); break;
            default: std::cout << "Unknown selection. Try 0-７.\n"; break; 
        }
    }
    return 0;
}
