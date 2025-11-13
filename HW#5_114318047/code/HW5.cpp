/********************************************************
* Filename    : HW5.cpp
* Author      : Stanley Chueh
* Note        : ADIP HW5 - menu-driven (portable)
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
void DFT(const char* input1, const char* input2, const char* output1, const char* output2) {
    // Open the input files
    FILE* file1 = fopen(input1, "rb");
    FILE* file2 = fopen(input2, "rb");

    if (!file1 || !file2) {
        std::cerr << "Error: Could not open input files.\n";
        if (file1) fclose(file1);
        if (file2) fclose(file2);
        return;
    }

    // Read the image data (assuming 256x256 grayscale images)
    const int width = 256, height = 256;
    uint8_t input_data[width * height];
    uint8_t input2_data[width * height];

    fread(input_data, sizeof(uint8_t), width * height, file1);
    fread(input2_data, sizeof(uint8_t), width * height, file2);

    fclose(file1);
    fclose(file2);

    std::cout << "Input images loaded successfully.\n";

    // Convert the image data to float arrays
    std::vector<std::vector<std::complex<float>>> input_float(height, std::vector<std::complex<float>>(width));
    std::vector<std::vector<std::complex<float>>> input2_float(height, std::vector<std::complex<float>>(width));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            input_float[i][j] = static_cast<float>(input_data[i * width + j]);
            input2_float[i][j] = static_cast<float>(input2_data[i * width + j]);
        }
    }

    std::cout << "Input images converted to float arrays.\n";

    // Perform DFT manually
    std::vector<std::vector<std::complex<float>>> input_dft(height, std::vector<std::complex<float>>(width));
    std::vector<std::vector<std::complex<float>>> input2_dft(height, std::vector<std::complex<float>>(width));

    const float pi = 3.14159265358979323846f;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            std::complex<float> sum1(0, 0);
            std::complex<float> sum2(0, 0);
            for (int x = 0; x < height; ++x) {
                for (int y = 0; y < width; ++y) {
                    float angle = -2.0f * pi * ((u * x / static_cast<float>(height)) + (v * y / static_cast<float>(width)));
                    std::complex<float> exp_term(cos(angle), sin(angle));
                    sum1 += input_float[x][y] * exp_term;
                    sum2 += input2_float[x][y] * exp_term;
                }
            }
            input_dft[u][v] = sum1;
            input2_dft[u][v] = sum2;
        }
    }

    std::cout << "DFT computation completed manually.\n";

    // Compute magnitude spectrum for visualization
    std::vector<std::vector<float>> input_magnitude(height, std::vector<float>(width));
    std::vector<std::vector<float>> input2_magnitude(height, std::vector<float>(width));

    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            input_magnitude[u][v] = std::log(1.0f + std::abs(input_dft[u][v])); // Log scale for better visualization
            input2_magnitude[u][v] = std::log(1.0f + std::abs(input2_dft[u][v]));
        }
    }

    // Normalize and save the magnitude spectrum as raw files
    FILE* output_file1 = fopen(output1, "wb");
    FILE* output_file2 = fopen(output2, "wb");

    if (!output_file1 || !output_file2) {
        std::cerr << "Error: Could not open output files.\n";
        if (output_file1) fclose(output_file1);
        if (output_file2) fclose(output_file2);
        return;
    }

    float max_val1 = 0, max_val2 = 0;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            max_val1 = std::max(max_val1, input_magnitude[u][v]);
            max_val2 = std::max(max_val2, input2_magnitude[u][v]);
        }
    }

    uint8_t input_magnitude_normalized[width * height];
    uint8_t input2_magnitude_normalized[width * height];

    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            input_magnitude_normalized[u * width + v] = static_cast<uint8_t>((input_magnitude[u][v] / max_val1) * 255);
            input2_magnitude_normalized[u * width + v] = static_cast<uint8_t>((input2_magnitude[u][v] / max_val2) * 255);
        }
    }

    fwrite(input_magnitude_normalized, sizeof(uint8_t), width * height, output_file1);
    fwrite(input2_magnitude_normalized, sizeof(uint8_t), width * height, output_file2);

    fclose(output_file1);
    fclose(output_file2);

    std::cout << "DFT and magnitude spectrum computation completed. Results saved as raw files.\n";
}

void DFT_OpenCV(const char* input1, const char* input2, const char* output1, const char* output2) {
    // Load the input images using OpenCV
    cv::Mat img1 = cv::imread(input1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(input2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load input images using OpenCV.\n";
        return;
    }

    std::cout << "Input images loaded successfully using OpenCV.\n";

    // Convert images to float type
    cv::Mat img1_float, img2_float;
    img1.convertTo(img1_float, CV_32F);
    img2.convertTo(img2_float, CV_32F);

    // Perform DFT using OpenCV
    cv::Mat dft1, dft2;
    cv::dft(img1_float, dft1, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(img2_float, dft2, cv::DFT_COMPLEX_OUTPUT);

    std::cout << "DFT computation completed using OpenCV.\n";

    // Compute magnitude spectrum
    cv::Mat magnitude1, magnitude2;
    cv::Mat planes1[] = {cv::Mat::zeros(dft1.size(), CV_32F), cv::Mat::zeros(dft1.size(), CV_32F)};
    cv::Mat planes2[] = {cv::Mat::zeros(dft2.size(), CV_32F), cv::Mat::zeros(dft2.size(), CV_32F)};
    cv::split(dft1, planes1);
    cv::split(dft2, planes2);

    cv::magnitude(planes1[0], planes1[1], magnitude1);
    cv::magnitude(planes2[0], planes2[1], magnitude2);

    // Switch to log scale for better visualization
    magnitude1 += cv::Scalar::all(1);
    magnitude2 += cv::Scalar::all(1);
    cv::log(magnitude1, magnitude1);
    cv::log(magnitude2, magnitude2);

    // Normalize the magnitude spectrum
    cv::normalize(magnitude1, magnitude1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(magnitude2, magnitude2, 0, 255, cv::NORM_MINMAX);

    // Convert to 8-bit for saving
    magnitude1.convertTo(magnitude1, CV_8U);
    magnitude2.convertTo(magnitude2, CV_8U);

    // Save the results
    if (!cv::imwrite(output1, magnitude1) || !cv::imwrite(output2, magnitude2)) {
        std::cerr << "Error: Could not save output images using OpenCV.\n";
        return;
    }

    // Save the output images as PNG files
    if (!cv::imwrite(output1, magnitude1)) {
        std::cerr << "Error: Could not save " << output1 << " as PNG.\n";
        return;
    }
    if (!cv::imwrite(output2, magnitude2)) {
        std::cerr << "Error: Could not save " << output2 << " as PNG.\n";
        return;
    }

    std::cout << "DFT and magnitude spectrum computation completed using OpenCV. Results saved as images.\n";
}
// 1. DFT
static void task1_1(const char* input_lena, const char* input_lena_noise, const char* output_lena, const char* output_lena_noise)
{
    DFT_OpenCV(input_lena, input_lena_noise, output_lena, output_lena_noise);

    std::cout << "task1_1 completed.\n";
}

int main() {
    std::cout << "----- Homework 5 Menu -----\n";
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
            std::cout << "Invalid input. Please enter a number between 0 and 12.\n";
            continue;
        }
        if (choice == 0) break;

        switch (choice) {
            case 1: task1_1("lena256.raw", "lena256_noise.raw", "lena256_out.raw", "lena256_noise_out.raw"); break;
            default: std::cout << "Unknown selection. Try 0-７.\n"; break; 
        }
    }
    return 0;
}
