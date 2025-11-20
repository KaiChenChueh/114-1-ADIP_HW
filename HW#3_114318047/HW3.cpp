/********************************************************
* Filename    : HW3.cpp
* Author      : Stanley Chueh
* Note        : ADIP HW3 - menu-driven (portable)
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
static void task_1_1(const char* input_image_dark_dog, const char* input_image_bright_dog,const char* input_image_low_contrast_dog,const char* output_image_dark_dog, const char* output_image_bright_dog,const char* output_image_low_contrast_dog) 
{
    const int W = 900, H = 600;
    FILE* input_file_dark = std::fopen(input_image_dark_dog, "rb");
    FILE* input_file_bright = std::fopen(input_image_bright_dog, "rb");
    FILE* input_file_low_contrast = std::fopen(input_image_low_contrast_dog, "rb");

    // Read the input image into a vector
    std::vector<unsigned char> image_original(W * H);

    std::vector<unsigned char> image_dark_dog = image_original;
    std::vector<unsigned char> image_bright_dog = image_original;
    std::vector<unsigned char> image_low_contrast_dog = image_original;
    
    std::fread(image_dark_dog.data(), 1, W * H, input_file_dark);
    std::fread(image_bright_dog.data(), 1, W * H, input_file_bright);
    std::fread(image_low_contrast_dog.data(), 1, W * H, input_file_low_contrast);

    std::fclose(input_file_dark);
    std::fclose(input_file_bright);
    std::fclose(input_file_low_contrast);

    // Power-Law Transformation for dark image(c=5, gamma=2.2)
    // s=c*(r^gamma)

    // dark dog
    int c_dark = 1;
    double gamma = 0.5;
    for (int r=0;r<H;++r) 
    {   
        uint8_t* rowPtr = image_dark_dog.data() + r * W; //image_oringinal.data() => pointer
        for (int c=0;c<W;++c)
        {
           uint8_t* pixelPtr = rowPtr + c;
            
            *pixelPtr = c_dark * std::pow(double(*pixelPtr) / 255.0, gamma) * 255.0;  // power-law transformation
        }                                                     // r should be normalized to [0,1],otherwise the result will be too large.
    }                                                         

    // bright dog
    int c_bright = 1;
    gamma = 2.2;
    for (int r=0;r<H;++r) 
    {   
        uint8_t* rowPtr = image_bright_dog.data() + r * W; //image_oringinal.data() => pointer
        for (int c=0;c<W;++c)
        {
           uint8_t* pixelPtr = rowPtr + c;
            
            *pixelPtr = c_bright * std::pow(double(*pixelPtr) / 255.0, gamma) * 255.0;  // power-law transformation
        }                                                     // r should be normalized to [0,1],otherwise the result will be too large.
    }

    //low contrast dog
    double c_low_contrast = 1.4;
    gamma = 2.4;
    for (int r=0;r<H;++r) 
    {   
        uint8_t* rowPtr = image_low_contrast_dog.data() + r * W; //image_oringinal.data() => pointer
        for (int c=0;c<W;++c)
        {
           uint8_t* pixelPtr = rowPtr + c;
            
            *pixelPtr = c_low_contrast * std::pow(double(*pixelPtr) / 255.0, gamma) * 255.0;  // power-law transformation
        }                                                     // r should be normalized to [0,1],otherwise the result will be too large.
    }

    // Write the shrunk image to the output file
    FILE* output_file_dark = std::fopen(output_image_dark_dog, "wb");
    FILE* output_file_bright = std::fopen(output_image_bright_dog, "wb");
    FILE* output_file_low_contrast = std::fopen(output_image_low_contrast_dog, "wb");

    std::fwrite(image_bright_dog.data(), 1, W * H , output_file_bright);
    std::fwrite(image_dark_dog.data(), 1, W * H , output_file_dark);
    std::fwrite(image_low_contrast_dog.data(), 1, W * H , output_file_low_contrast);

    std::fclose(output_file_dark);
    std::fclose(output_file_bright);
    std::fclose(output_file_low_contrast);
}

// Piecewise-Linear Transformation
static void task_1_2(const char* input_image_dark_dog, const char* input_image_bright_dog,const char* input_image_low_contrast_dog,const char* output_image_dark_dog, const char* output_image_bright_dog,const char* output_image_low_contrast_dog) 
{
    const int W = 900, H = 600;
    FILE* input_file_dark = std::fopen(input_image_dark_dog, "rb");
    FILE* input_file_bright = std::fopen(input_image_bright_dog, "rb");
    FILE* input_file_low_contrast = std::fopen(input_image_low_contrast_dog, "rb");

    // Read the input image into a vector
    std::vector<unsigned char> image_original(W * H);

    std::vector<unsigned char> image_dark_dog = image_original;
    std::vector<unsigned char> image_bright_dog = image_original;
    std::vector<unsigned char> image_low_contrast_dog = image_original;
    
    std::fread(image_dark_dog.data(), 1, W * H, input_file_dark);
    std::fread(image_bright_dog.data(), 1, W * H, input_file_bright);
    std::fread(image_low_contrast_dog.data(), 1, W * H, input_file_low_contrast);

    std::fclose(input_file_dark);
    std::fclose(input_file_bright);
    std::fclose(input_file_low_contrast);

    // Piecewise-Linear Transformation for dark images

    // dark dog 
    std::vector<std::pair<int, int>> dark_points = 
    { 
        {0, 0}, {30, 60}, {70, 120}, {100, 160}, {140, 200}, {200, 220}, {255, 255} 
    };
    std::sort(dark_points.begin(), dark_points.end(),
            [](auto &a, auto &b){ return a.first < b.first; });

    for (int r = 0; r < H; ++r) {
        uint8_t* rowPtr = image_dark_dog.data() + r * W;
        for (int c = 0; c < W; ++c) {
            uint8_t* pixelPtr = rowPtr + c;
            int v = *pixelPtr;  // original pixel values

            // find segment
            for (size_t i = 0; i + 1 < dark_points.size(); ++i) {
                int x0 = dark_points[i].first;
                int y0 = dark_points[i].second;
                int x1 = dark_points[i + 1].first;
                int y1 = dark_points[i + 1].second;

                /*  (x1,y1)                    /
                |   __________________________/
                |  /
                | /v
                |/____________________
                (x0,y0)
                */
                if (v >= x0 && v <= x1) // check which segment(v)
                {                       
                    int out;
                    if (x1 - x0 == 0) 
                    {
                        out = y0;  // degenerate segment
                    } 
                    else {
                        double y = y0 + double(y1 - y0) * double(v - x0) / double(x1 - x0); // y=mx+b, 
                        out = int(std::lround(std::clamp(y, 0.0, 255.0)));                  // m = (y1 - y0) / (x1 - x0)
                        // the result of std::lround is a long type, so we need to cast it to int
                        // std::clamp to ensure the value is within [0,255]
                    }
                    // the data type of out is int, convert to uint8_t to store in the image
                    *pixelPtr = uint8_t(out);
                    break;
                }
            }
        }
    }                                                    

    // bright dog
    std::vector<std::pair<int, int>> bright_points = 
    { {0, 0},    {50, 20},   {100, 50}, 
      {150, 120}, {200, 160}, {255, 200} 
    };
    std::sort(bright_points.begin(), bright_points.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
    for (int r = 0; r < H; ++r) {
        uint8_t* rowPtr = image_bright_dog.data() + r * W;
        for (int c = 0; c < W; ++c) {
            uint8_t* pixelPtr = rowPtr + c;
            int v = *pixelPtr;  // original pixel values

            // find segment
            for (size_t i = 0; i + 1 < bright_points.size(); ++i) {
                int x0 = bright_points[i].first;
                int y0 = bright_points[i].second;
                int x1 = bright_points[i + 1].first;
                int y1 = bright_points[i + 1].second;

                if (v >= x0 && v <= x1) // check which segment(v)
                {                       
                    int out;
                    if (x1 - x0 == 0) 
                    {
                        out = y0;  
                    } 
                    else {
                        double y = y0 + double(y1 - y0) * double(v - x0) / double(x1 - x0); // y=mx+b, 
                        out = int(std::lround(std::clamp(y, 0.0, 255.0)));                  // m = (y1 - y0) / (x1 - x0)
                    }
                    *pixelPtr = uint8_t(out);
                    break;
                }
            }
        }
    }

    // low contrast dog
    std::vector<std::pair<int, int>> low_contrast_points = 
    { {0, 0},    {50, 20},   {100, 60}, 
      {150, 180}, {200, 240}, {255, 255} 
    };
    std::sort(low_contrast_points.begin(), low_contrast_points.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
    for (int r = 0; r < H; ++r) {
        uint8_t* rowPtr = image_low_contrast_dog.data() + r * W;
        for (int c = 0; c < W; ++c) {
            uint8_t* pixelPtr = rowPtr + c;
            int v = *pixelPtr;  // original pixel values

            // find segment
            for (size_t i = 0; i + 1 < low_contrast_points.size(); ++i) {
                int x0 = low_contrast_points[i].first;
                int y0 = low_contrast_points[i].second;
                int x1 = low_contrast_points[i + 1].first;
                int y1 = low_contrast_points[i + 1].second;

                if (v >= x0 && v <= x1) // check which segment(v)
                {                       
                    int out;
                    if (x1 - x0 == 0) 
                    {
                        out = y0;  // degenerate segment
                    } 
                    else {
                        double y = y0 + double(y1 - y0) * double(v - x0) / double(x1 - x0); // y=mx+b, 
                        out = int(std::lround(std::clamp(y, 0.0, 255.0)));                  // m = (y1 - y0) / (x1 - x0)
                    }
                    *pixelPtr = uint8_t(out);
                    break;
                }
            }
        }
    }

    // Write the shrunk image to the output file
    FILE* output_file_dark = std::fopen(output_image_dark_dog, "wb");
    FILE* output_file_bright = std::fopen(output_image_bright_dog, "wb");
    FILE* output_file_low_contrast = std::fopen(output_image_low_contrast_dog, "wb");

    std::fwrite(image_bright_dog.data(), 1, W * H , output_file_bright);
    std::fwrite(image_dark_dog.data(), 1, W * H , output_file_dark);
    std::fwrite(image_low_contrast_dog.data(), 1, W * H , output_file_low_contrast);

    std::fclose(output_file_dark);
    std::fclose(output_file_bright);
    std::fclose(output_file_low_contrast);
}

// Use XnView to binarize fire flower and show
static void task_2_1() 
{
    std::cout << "Please check the report for the fire flower image in 512x512(binary).\n";
}

// Replace bit-plane 3,5,7 of lena512.raw one by one with the binarized image in (1)
// and show the results.After that,extract the replaced bit-planed from it to prove and ensure the fire flower
// image is correctly implanted.
static void task_2_2(const char* input_image_lena, const char* input_image_fire_flower)
{
    int W = 512, H = 512;

    // Read lena512.raw and fire_flower_binarized.raw
    FILE* input_file_lena = std::fopen(input_image_lena, "rb");
    FILE* input_file_fire_flower = std::fopen(input_image_fire_flower, "rb");

    std::vector<unsigned char> image_lena(W * H);
    std::vector<unsigned char> image_fire_flower(W * H);

    std::fread(image_lena.data(), 1, W * H, input_file_lena);
    std::fread(image_fire_flower.data(), 1, W * H, input_file_fire_flower);
   
    std::fclose(input_file_lena);
    std::fclose(input_file_fire_flower);

    // Replace bit-plane 3, 5, 7 of lena512.raw one by one with the binary image
    for (int bp = 3; bp <= 7; bp += 2) { // Process bit-planes 3, 5, and 7
        std::vector<unsigned char> image_lena_copy = image_lena;
        for (int i = 0; i < W * H; ++i) {
            unsigned char lena_pixel = image_lena_copy[i];
            unsigned char fire_pixel = image_fire_flower[i]; // Binary 
            unsigned char bit = (image_fire_flower[i] != 0) ? 1 : 0; // Force to 0 or 1
            lena_pixel = (lena_pixel & ~(1 << bp)) | (bit << bp);
            // 1<<bp , shift 1 to left for bp times=> create a mask for the target bit-plane(bp=3=>00001000), ~(1<<bp) => invert the mask(11110111)
            // and do the AND operation to clear the target bit-plane，　then do the OR operation to set the target bit-plane to the new value
            // bit << bp=> shift the bit to the target bit-plane position

            image_lena_copy[i] = lena_pixel;
        }

        // Write the modified Lena image to an output file for the current bit-plane
        std::string output_filename = "lena512_modified_bitplane_" + std::to_string(bp) + ".raw";
        FILE* output_file = std::fopen(output_filename.c_str(), "wb");
        std::fwrite(image_lena_copy.data(), 1, W * H, output_file);
        std::fclose(output_file);

        // Proof
        std::vector<unsigned char> proof(W * H);
        for (int i = 0; i < W * H; ++i)
            proof[i] = ((image_lena_copy[i] >> bp) & 1) ? 255 : 0; //set to 255 for white and 0 for black
        FILE* f = std::fopen(("extract_bp" + std::to_string(bp) + ".raw").c_str(), "wb");
        std::fwrite(proof.data(), 1, W*H, f);
        std::fclose(f);

        // Calculate MSE
        double mse = 0.0;
        for (int i = 0; i < W * H; ++i) {
            double diff = double((image_lena[i])) - double((image_lena_copy[i]));
            mse += diff * diff;
        }
        mse /= (W * H);

        std::cout << "Saved output for bit-plane " << bp << " to " << output_filename << "\n";
        std::cout << "MSE for bit-plane " << bp << ": " << mse << "\n";
    }
}

// Histogram
// plot the histogram of dog_dark_900x600.raw, dog_bright_900x600.raw
static void task_3_1(const char* input_image_dark_dog, const char* input_image_bright_dog)
{
    int W = 900, H = 600;
    FILE* input_file_dark = std::fopen(input_image_dark_dog, "rb");
    FILE* input_file_bright = std::fopen(input_image_bright_dog, "rb");

    std::vector<unsigned char> image_dark_dog(W * H);
    std::vector<unsigned char> image_bright_dog(W * H);

    std::fread(image_dark_dog.data(), 1, W * H, input_file_dark);
    std::fread(image_bright_dog.data(), 1, W * H, input_file_bright);

    std::fclose(input_file_dark);
    std::fclose(input_file_bright);

    // Compute histogram
    std::vector<int> histogram_dark(256, 0);
    std::vector<int> histogram_bright(256, 0); // 256 intensity levels

    // dog_dark
    for (int k = 0; k < W * H; ++k)
    {
        histogram_dark[image_dark_dog[k]]++;
    }

    // dog_bright
    for (int k = 0; k < W * H; ++k)
    {
        histogram_bright[image_bright_dog[k]]++;
    }
        
    // Plot histogram using OpenCV
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w / 256);

    cv::Mat histImage_dark(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    cv::Mat histImage_bright(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

    // Normalize the result to [0, histImage.rows]
    // std::max_element returns an address of the max element,so add *
    int max_dark = *std::max_element(histogram_dark.begin(), histogram_dark.end());
    int max_bright = *std::max_element(histogram_bright.begin(), histogram_bright.end());

    // Calculate the height of each bin
    for (int i = 0; i < 256; i++) {
        histogram_dark[i] = ((double)histogram_dark[i] / max_dark) * hist_h;
        histogram_bright[i] = ((double)histogram_bright[i] / max_bright) * hist_h;
    }

    // Draw for each channel
    for (int i = 1; i < 256; i++) {
        cv::line(histImage_dark, cv::Point(bin_w * (i - 1), hist_h - histogram_dark[i - 1]),
                 cv::Point(bin_w * (i), hist_h - histogram_dark[i]),
                 cv::Scalar(255), 2, 8, 0);
        cv::line(histImage_bright, cv::Point(bin_w * (i - 1), hist_h - histogram_bright[i - 1]),
                 cv::Point(bin_w * (i), hist_h - histogram_bright[i]),
                 cv::Scalar(255), 2, 8, 0);
    }
    // Display
    cv::imshow("Histogram of Dark Dog Image", histImage_dark);
    cv::imshow("Histogram of Bright Dog Image", histImage_bright);
    cv::waitKey(0);
}

// Perform histogram equalization on dog_dark_900x600.raw, dog_bright_900x600.raw
static void task_3_2(const char* input_image_dark_dog, const char* input_image_bright_dog,const char* output_image_dark_dog, const char* output_image_bright_dog)
{
    int W = 900, H = 600;

    FILE* input_file_dark = std::fopen(input_image_dark_dog, "rb");
    FILE* input_file_bright = std::fopen(input_image_bright_dog, "rb");

    std::vector<unsigned char> image_dark_dog(W * H);
    std::vector<unsigned char> image_bright_dog(W * H);

    std::fread(image_dark_dog.data(), 1, W * H, input_file_dark);
    std::fread(image_bright_dog.data(), 1, W * H, input_file_bright);

    std::fclose(input_file_dark);
    std::fclose(input_file_bright);

    // Histogram Equalization for dark dog image and bright dog image
    std::vector<int> histogram_dark(256, 0);
    std::vector<int> histogram_bright(256, 0);

    // Compute histogram
    for (int i = 0; i < W * H; ++i) {
        histogram_dark[image_dark_dog[i]]++;
    }

    for (int i = 0; i < W * H; ++i) {
        histogram_bright[image_bright_dog[i]]++;
    }

    // Compute CDF
    std::vector<int> cdf_dark(256, 0);
    std::vector<int> cdf_bright(256, 0);

    cdf_dark[0] = histogram_dark[0];
    for (int i = 1; i < 256; ++i) {
        cdf_dark[i] = cdf_dark[i - 1] + histogram_dark[i];
    }

    cdf_bright[0] = histogram_bright[0];
    for (int i = 1; i < 256; ++i) {
        cdf_bright[i] = cdf_bright[i - 1] + histogram_bright[i];
    }

    // Normalize CDF
    int cdf_min_dark = 0; int cdf_min_bright = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf_dark[i] != 0) {
            cdf_min_dark = cdf_dark[i];
            break; // find the first non-zero value
        }
    }

    for (int i = 0; i < 256; ++i) {
        if (cdf_bright[i] != 0) {
            cdf_min_bright = cdf_bright[i];
            break; // find the first non-zero value
        }
    }

    // Map the pixel values using the normalized CDF
    std::vector<unsigned char> equalized_dark(W * H);
    std::vector<unsigned char> equalized_bright(W * H);
    /*
    Histrogram Equalization formula:
    1. Find histogram
    2. Find PDF => Find CDF
    3. Normalize CDF:(L-1)*cdf => rounding, L=>8 bit(maps to 0-255)
    4. Map the pixel values
    */
    for (int i = 0; i < W * H; ++i) {
        equalized_dark[i] = std::round((double)(cdf_dark[image_dark_dog[i]] - cdf_min_dark) / (W * H - cdf_min_dark) * 255.0);
        // cdf_dark[image_dark_dog[i]] => get the cdf value of the pixel
        // cdf_dark is not cdf, it's the cumulative frequency, so we need to normalize it(divide by total number of pixels)
        // (W*H - cdf_min_dark) => total number of pixels 
    }
    for (int i = 0; i < W * H; ++i) {
        equalized_bright[i] = std::round((double)(cdf_bright[image_bright_dog[i]] - cdf_min_bright) / (W * H - cdf_min_bright) * 255.0);
    }

    // Display histogram equalized results
    // Compute histogram for equalized dark image
    std::vector<int> histogram_equalized_dark(256, 0);
    for (int i = 0; i < W * H; ++i) {
        histogram_equalized_dark[equalized_dark[i]]++;
    }

    // Compute histogram for equalized bright image
    std::vector<int> histogram_equalized_bright(256, 0);
    for (int i = 0; i < W * H; ++i) {
        histogram_equalized_bright[equalized_bright[i]]++;
    }

    // Plot histogram using OpenCV
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w / 256);

    cv::Mat histImage_equalized_dark(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    cv::Mat histImage_equalized_bright(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

    // Normalize the result to [0, histImage.rows]
    int max_equalized_dark = *std::max_element(histogram_equalized_dark.begin(), histogram_equalized_dark.end());
    int max_equalized_bright = *std::max_element(histogram_equalized_bright.begin(), histogram_equalized_bright.end());
    for (int i = 0; i < 256; i++) {
        histogram_equalized_dark[i] = ((double)histogram_equalized_dark[i] / max_equalized_dark) * hist_h;
        histogram_equalized_bright[i] = ((double)histogram_equalized_bright[i] / max_equalized_bright) * hist_h;
    }

    // Draw for each channel
    for (int i = 1; i < 256; i++) {
        cv::line(histImage_equalized_dark, cv::Point(bin_w * (i - 1), hist_h - histogram_equalized_dark[i - 1]),
                 cv::Point(bin_w * (i), hist_h - histogram_equalized_dark[i]),
                 cv::Scalar(255), 2, 8, 0);
        cv::line(histImage_equalized_bright, cv::Point(bin_w * (i - 1), hist_h - histogram_equalized_bright[i - 1]),
                 cv::Point(bin_w * (i), hist_h - histogram_equalized_bright[i]),
                 cv::Scalar(255), 2, 8, 0);
    }

    // Display
    cv::imshow("Histogram of Equalized Dark Dog Image", histImage_equalized_dark);
    cv::imshow("Histogram of Equalized Bright Dog Image", histImage_equalized_bright);
    cv::waitKey(0);
    
}

// Perform histogram matching on lena512.raw by matching dog_bright_900x600.raw, Plot the histogram after matching
static void task_3_3(const char* input_image_lena, const char* input_image_bright_dog)
{
    int W = 512, H = 512; // lena
    int W_dog = 900, H_dog = 600; // dog_bright

    // Read lena512.raw and dog_bright_900x600.raw
    FILE* input_file_lena = std::fopen(input_image_lena, "rb");
    FILE* input_file_bright_dog = std::fopen(input_image_bright_dog, "rb");

    std::vector<unsigned char> image_lena(W * H);
    std::vector<unsigned char> image_bright_dog(W_dog * H_dog);

    std::fread(image_lena.data(), 1, W * H, input_file_lena);
    std::fread(image_bright_dog.data(), 1, W_dog * H_dog, input_file_bright_dog);

    std::fclose(input_file_lena);
    std::fclose(input_file_bright_dog);

    // Compute histogram and CDF for lena image
    std::vector<int> histogram_lena(256, 0);
    for (int i = 0; i < W * H; ++i) {
        histogram_lena[image_lena[i]]++;
    }

    std::vector<int> cdf_lena(256, 0);
    cdf_lena[0] = histogram_lena[0];
    for (int i = 1; i < 256; ++i) {
        cdf_lena[i] = cdf_lena[i - 1] + histogram_lena[i];
    }

    // Compute histogram and CDF for dog_bright image
    std::vector<int> histogram_bright_dog(256, 0);
    for (int i = 0; i < W_dog * H_dog; ++i) {
        histogram_bright_dog[image_bright_dog[i]]++;
    }

    std::vector<int> cdf_bright_dog(256, 0);
    cdf_bright_dog[0] = histogram_bright_dog[0];
    for (int i = 1; i < 256; ++i) {
        cdf_bright_dog[i] = cdf_bright_dog[i - 1] + histogram_bright_dog[i];
    }

    // Normalize CDFs cdf_lena now in the code is not normalized
    for (int i = 0; i < 256; ++i) {
        cdf_lena[i] = (cdf_lena[i] * 255) / (W * H); // Normalize using total number of pixels
        cdf_bright_dog[i] = (cdf_bright_dog[i] * 255) / (W_dog * H_dog); // Normalize using total number of pixels in dog image
    }

    // Create a mapping from lena intensity values to dog_bright intensity values
    std::vector<unsigned char> mapping(256, 0);
    int j = 0;
    for (int i = 0; i < 256; ++i) {
        while (j < 256 && cdf_bright_dog[j] < cdf_lena[i]) {
            j++; // find above or equal to
        }
        mapping[i] = j;
    }
    // Lena
    /*
    gray level       Number of pixels         CDF
    0                       10                0.1
    1                       30                0.4
    2                       35                0.75
    3                       25                1.0

    // Dog
    gray level       Number of pixels         CDF
    0                       5                 0.05
    1                       45                0.5
    2                       40                0.9
    3                       10                1.0

    // LUT mapping
    Lena gray level   Lena CDF    Closest Dog Gray Level(greater than below or equal to)
        0                 0.1              1     
        1                 0.4              1
        2                 0.75             2
        3                 1.0              3

    Original lena:[0,1,2,3] => Mapped lena:[1,1,2,3]
    */

    // Apply the mapping to lena image
    std::vector<unsigned char> matched_lena(W * H);
    for (int i = 0; i < W * H; ++i) {
        matched_lena[i] = mapping[image_lena[i]];
    }

    // Compute histogram for matched lena image
    std::vector<int> histogram_matched_lena(256, 0);
    for (int i = 0; i < W * H; ++i) {
        histogram_matched_lena[matched_lena[i]]++;
    }

    // Plot histogram for matched lena image
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    cv::Mat histImage_matched_lena(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    int max_matched_lena = *std::max_element(histogram_matched_lena.begin(), histogram_matched_lena.end());
    for (int i = 0; i < 256; i++) {
        histogram_matched_lena[i] = ((double)histogram_matched_lena[i] / max_matched_lena) * hist_h;
    }
    for (int i = 1; i < 256; i++) {
        cv::line(histImage_matched_lena, cv::Point(bin_w * (i - 1), hist_h - histogram_matched_lena[i - 1]),
                 cv::Point(bin_w * (i), hist_h - histogram_matched_lena[i]),
                 cv::Scalar(255), 2, 8, 0);
    }

    // Display histograms
    cv::imshow("Histogram of Matched Lena Image", histImage_matched_lena);

    // Display the processed lena image
    cv::Mat lena_matched(H, W, CV_8UC1, matched_lena.data());
    cv::imshow("Processed Lena Image", lena_matched);

    cv::waitKey(0);

    // Save the matched lena image
    FILE* output_file_matched_lena = std::fopen("lena512_matched.raw", "wb");
    std::fwrite(matched_lena.data(), 1, W * H, output_file_matched_lena);
    std::fclose(output_file_matched_lena);
}
int main() {
    std::cout << "----- Homework 3 Menu -----\n";
    while (true) {
        std::cout << "\n================ Results Menu ================\n"
                  << " 1) 1-1\n"
                  << " 2) 1-2\n"
                  << " 3) 2-1\n"
                  << " 4) 2-2\n"
                  << " 5) 3-1\n"
                  << " 6) 3-2\n"
                  << " 7) 3-3\n"
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
            case 1: task_1_1("dog_dark_900x600.raw","dog_bright_900x600.raw","dog_low_contrast_900x600.raw","dog_dark_900x600_output.raw","dog_bright_900x600_output.raw","dog_low_constrast_900x600_output.raw"); break;
            case 2: task_1_2("dog_dark_900x600.raw","dog_bright_900x600.raw","dog_low_contrast_900x600.raw","dog_dark_900x600_output_1_2.raw","dog_bright_900x600_output_1_2.raw","dog_low_constrast_900x600_output_1_2.raw"); break;
            case 3: task_2_1(); break;
            case 4: task_2_2("lena512.raw","fire_flower_binarized.raw"); break;
            case 5: task_3_1("dog_dark_900x600.raw","dog_bright_900x600.raw"); break;
            case 6: task_3_2("dog_dark_900x600.raw","dog_bright_900x600.raw","dog_dark_900x600_equalized.raw","dog_bright_900x600_equalized.raw"); break;
            case 7: task_3_3("lena512.raw","dog_bright_900x600.raw"); break;
            default: std::cout << "Unknown selection. Try 0-７.\n"; break; 
        }
    }
    return 0;
}
