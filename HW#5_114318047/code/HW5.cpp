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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helpers
// DFT
void DFT_1D(const std::vector<std::complex<float>>& f_x, std::vector<std::complex<float>>& f_u, int M)
{
    // Discrete DFT formula
    /*
        F(u) = 1 / M * Σ(x=0 to M-1) f(x) * e^(-j*2π*u*x / M) u=0,1,...,M-1

        e^(-j*θ) = cos(θ) + j*sin(θ)
        e^(-j*2π*u*x / M) = cos(-2π*u*x / M) + j*sin(-2π*u*x / M)
    */
    // Do DFT to row vector
    for (int u = 0; u < M; u++) {
        std::complex<float> sum(0,0);

        for (int x = 0; x < M; x++) {
            float angle = -2.0f * M_PI * u * x / M;
            sum += f_x[x] * std::complex<float>(cos(angle), sin(angle));
        }
        f_u[u] = sum / (float)M;
    }
}

void DFT_2D(const std::vector<std::vector<float>>& f_xy, std::vector<std::vector<std::complex<float>>>& f_uv, int M, int N)
{   
    // 2-D DFT(Width and Height)
    /*
        F(u,v) = 1 / (M*N) * Σ(x=0 to M-1) Σ(y=0 to N-1) f(x,y) * e^(-j*2π*(u*x/M + v*y/N)) u=0,1,...,M-1; v=0,1,...,N-1

        e^(-j*θ) = cos(θ) + j*sin(θ)
        e^(-j*2π*(u*x/M + v*y/N)) = cos(-2π*(u*x/M + v*y/N)) + j*sin(-2π*(u*x/M + v*y/N))
    */

    // Step 1: origin shift (-1)^(x+y)
    /*
        f(x,y)*e^(j*2π*(u0*x/M + v0*y/N)) = F(u-u0, v-v0)
        
        when u0 = M/2, v0 = N/2
        e^(-j*2π*(u0*x/M + v0*y/N)) = e^(-j*2π*(M/2*x/M + N/2*y/N)) = e^(-j*π*(x + y)) = (-1)^(x+y)
    */

    // Create a shifted image matrix MxN
    // std::vector<std::complex<float>> => 1-D complex vector: [j0, j1, j2, ..., jN-1]
    // std::vector< std::vector<std::complex<float>> > => 2-D complex matrix: [ [j00, j01, ..., j0N-1],
    // shifted(M, std::vector<std::complex<float>>(N)) => create M rows, each row has N complex elements

    std::vector< std::vector<std::complex<float>> > shifted(M, std::vector<std::complex<float>>(N));
    for (int x = 0; x < M; x++)
        for (int y = 0; y < N; y++)
            // (-1)^(x+y) => (((x + y) % 2 == 0) ? 1.0f : -1.0f)
            shifted[x][y] = f_xy[x][y] * (((x + y) % 2 == 0) ? 1.0f : -1.0f);

    // Step 2: DFT for each row (N)
    std::vector<std::vector<std::complex<float>>> rowDFT(M, std::vector<std::complex<float>>(N));
    for (int x = 0; x < M; x++) {
        DFT_1D(shifted[x], rowDFT[x], N);
    }

    // Step 3: DFT for each column (M)
    for (int y = 0; y < N; y++)
    {
        // col_in: 
        std::vector<std::complex<float>> col_in(M), col_out(M);

        // extract column
        /*
            column is not continuous in memory, so we need to extract it
            first row:
            col_in[0] = rowDFT[0][y] 
            second row:
            col_in[1] = rowDFT[1][y]
            .....
            M-th row:
            col_in[M-1] = rowDFT[M-1][y]
            Now, col_in holds the y-th column of the rowDFT
        〞
        */
        for (int x = 0; x < M; x++)
            col_in[x] = rowDFT[x][y];

        // 1D DFT on column
        DFT_1D(col_in, col_out, M);

        // place into output
        for (int x = 0; x < M; x++)
            f_uv[x][y] = col_out[x];
    }
}

// IDFT
void IDFT_1D(const std::vector<std::complex<float>>& f_u, std::vector<std::complex<float>>& f_x, int M)
{
    // Discrete IDFT formula
    /*
        F(u) = Σ(u=0 to M-1) f(x) * e^(j*2π*u*x / M) x=0,1,...,M-1

        e^(j*θ) = cos(θ) + j*sin(θ)
        e^(j*2π*u*x / M) = cos(2π*u*x / M) + j*sin(2π*u*x / M)
    */
    for (int x = 0; x < M; x++) {
        std::complex<float> sum(0,0);

        for (int u = 0; u < M; u++) {
            float angle = 2.0f * M_PI * u * x / M;
            sum += f_u[u] * std::complex<float>(cos(angle), sin(angle));
        }
        f_x[x] = sum;
    }
}

void IDFT_2D(const std::vector<std::vector<std::complex<float>>>& f_uv, std::vector<std::vector<float>>& f_xy, int M, int N)
{
    // 2-D IDFT(Width and Height)
    /*
        F(x,y) = Σ(u=0 to M-1) Σ(v=0 to N-1) f(u,v) * e^(j*2π*(u*x/M + v*y/N)) x=0,1,...,M-1; y=0,1,...,N-1

        e^(-j*θ) = cos(θ) + j*sin(θ)
        e^(-j*2π*(u*x/M + v*y/N)) = cos(-2π*(u*x/M + v*y/N)) + j*sin(-2π*(u*x/M + v*y/N))
    */
    // Step 1: IDFT for each column
    std::vector<std::vector<std::complex<float>>> colIDFT(M, std::vector<std::complex<float>>(N));

    for (int v = 0; v < N; v++) {

        std::vector<std::complex<float>> col_in(M), col_out(M);

        // extract column v
        for (int u = 0; u < M; u++)
            col_in[u] = f_uv[u][v];

        // IDFT along column
        IDFT_1D(col_in, col_out, M);

        // store result
        for (int x = 0; x < M; x++)
            colIDFT[x][v] = col_out[x];
    }

    // Step 2: IDFT for each row
    std::vector<std::vector<std::complex<float>>> rowIDFT(M, std::vector<std::complex<float>>(N));

    for (int x = 0; x < M; x++)
    {
        std::vector<std::complex<float>> row_in(N), row_out(N);

        // extract row x
        for (int v = 0; v < N; v++)
            row_in[v] = colIDFT[x][v];

        // IDFT along row
        IDFT_1D(row_in, row_out, N);

        // store
        for (int y = 0; y < N; y++)
            rowIDFT[x][y] = row_out[y];
    }

    // Step 3: origin shift + real part
    for (int x = 0; x < M; x++)
        for (int y = 0; y < N; y++)
            f_xy[x][y] = std::real(rowIDFT[x][y]) * (((x + y) % 2 == 0) ? 1.0f : -1.0f);
}

// PSNR for comparing restored image and original image
double computePSNR(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2)
{
    const int N = img1.size();
    double mse = 0.0;

    // PSNR
    /*
        10 * log10((MAX_I^2) / MSE)
    */

    // find MSE
    for (int i = 0; i < N; i++) {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }

    // average
    mse /= N;

    if (mse == 0)
        return INFINITY;   // identical images


    // Do PSNR
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    return psnr;
}

// Butterworth filter
void Apply_butterworth_filter(cv::Mat& filtered_dft, const cv::Mat& Hmask, const cv::Mat *planes, int H, int W)
{
    filtered_dft = cv::Mat(H, W, CV_32FC2);

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            // planes[0]: real part
            // planes[1]: imaginary part
            // (real, imag) * H(u,v)
            float H = Hmask.at<float>(i, j);
            float real = planes[0].at<float>(i, j);
            float imag = planes[1].at<float>(i, j);

            filtered_dft.at<cv::Vec2f>(i, j)[0] = real * H;
            filtered_dft.at<cv::Vec2f>(i, j)[1] = imag * H;
        }
}

void gaussianBlur3x3(const cv::Mat& src, cv::Mat& dst)
{
    // Ensure the input image is of type CV_32F (32-bit float single channel)
    if (src.type() != CV_32F) {
        throw std::invalid_argument("Input image must be of type CV_32F (32-bit float).");
    }

    int H = src.rows;
    int W = src.cols;

    dst = cv::Mat::zeros(H, W, CV_32F);

    // Gaussian kernel
    static int G[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            float sum = 0.0f;
            float norm = 16.0f;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    // yy, xx: handle border by clamping
                    int yy = std::min(std::max(y + ky, 0), H - 1);
                    int xx = std::min(std::max(x + kx, 0), W - 1);

                    sum += src.at<float>(yy, xx) * G[ky + 1][kx + 1];
                }
            }

            dst.at<float>(y, x) = sum / norm;
        }
    }
}

// Helper function to process and save magnitude spectrum
void processAndSaveMagnitude(const cv::Mat& dft, const std::string& output_filename) {
    cv::Mat planes[2];
    cv::split(dft, planes);

    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    mag += 1;
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    cv::imwrite(output_filename, mag);
}

// 1. DFT
static void task1_1(const char* input_lena, const char* input_lena_noise,
                    std::vector<std::vector<std::complex<float>>>& dft_out,
                    std::vector<std::vector<std::complex<float>>>& dft_noise_out)
{
    // Call by reference to return dft_out and dft_noise_out for task1_2

    const int H = 256, W = 256;

    std::vector<uint8_t> input_lena_buffer(H * W), input_lena_noise_buffer(H * W);

    FILE* f1 = fopen(input_lena, "rb");
    FILE* f2 = fopen(input_lena_noise, "rb");

    fread(input_lena_buffer.data(), 1, H * W, f1);
    fread(input_lena_noise_buffer.data(), 1, H * W, f2);

    fclose(f1);
    fclose(f2);

    // Execution time measurement
    auto start =  std::chrono::high_resolution_clock::now();

    // Convert to float
    // img(H, std::vector<float>(W)) => create H rows, each row has W float elements
    std::vector<std::vector<float>> img(H, std::vector<float>(W));
    std::vector<std::vector<float>> img_noise(H, std::vector<float>(W));

    // Copy data from 1-D buffer to 2-D image matrix
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            img[i][j] = input_lena_buffer[i*W + j];
            img_noise[i][j] = input_lena_noise_buffer[i*W + j];
        }

    // Do 2-D DFT
    // Alocate

    dft_out = std::vector<std::vector<std::complex<float>>>(H, std::vector<std::complex<float>>(W));
    dft_noise_out = std::vector<std::vector<std::complex<float>>>(H, std::vector<std::complex<float>>(W));

    DFT_2D(img, dft_out, H, W);
    DFT_2D(img_noise, dft_noise_out, H, W);

    // Compute magnitude spectrum
    std::vector<float> mag(H*W), mag_noise(H*W);

    // Find min/max for normalization
    float min1=1e9, max1=-1e9, min2=1e9, max2=-1e9;

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {

            // std::abs => abs(a+jb) = sqrt(a^2 + b^2)(magnitude)
            // log(1 + magnitude) for visualization
            float v1 = logf(1 + std::abs(dft_out[i][j])) * 10.0f;
            float v2 = logf(1 + std::abs(dft_noise_out[i][j])) * 10.0f;

            mag[i*W + j] = v1;
            mag_noise[i*W + j] = v2;

            min1 = std::min(min1, v1);
            max1 = std::max(max1, v1);
            min2 = std::min(min2, v2);
            max2 = std::max(max2, v2);
        }

    std::vector<uint8_t> outmag(H * W), outmag2(H * W);

    // Normalize to [0,255] for visualization
    // https://ithelp.ithome.com.tw/articles/10293893
    /*
        z = 255 * (x - min) / (max - min)
    */

    // Gamma correction for better visualization
    float gamma = 0.2f;   
    for (int k = 0; k < H * W; k++) {
        float n1 = (mag[k] - min1) / (max1 - min1);         // normalization
        float n2 = powf(n1, gamma);                         // gamma correction
        outmag[k]  = (uint8_t)(255.0f * n2);

        float n1n = (mag_noise[k] - min2) / (max2 - min2);
        float n2n = powf(n1n, gamma);
        outmag2[k] = (uint8_t)(255.0f * n2n);
    }

    // save RAW
    const char* out_mag = "lena_dft_mag.raw";
    const char* out_mag_noise = "lena_noise_dft_mag.raw";

    FILE* fm1 = fopen(out_mag, "wb");
    FILE* fm2 = fopen(out_mag_noise, "wb");

    fwrite(outmag.data(), 1, H*W, fm1);
    fwrite(outmag2.data(), 1, H*W, fm2);

    fclose(fm1); 
    fclose(fm2);

    // save as png
    cv::Mat mag_img(H, W, CV_8UC1, outmag.data());  
    cv::Mat mag_img_noise(H, W, CV_8UC1, outmag2.data());

    cv::imwrite(std::string(out_mag) + ".png", mag_img);
    cv::imwrite(std::string(out_mag_noise) + ".png", mag_img_noise);

    // Ends time measurement
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "task1_1 done: magnitude saved.\n";
    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Custom DFT time: " << duration << " ms\n";
}

// IDFT
static void task1_2(const std::vector<std::vector<std::complex<float>>>& dft_lena, const std::vector<std::vector<std::complex<float>>>& dft_lena_noise,
                    const char* out_restore, const char* out_restore_noise)
{
    const int H = 256, W = 256;

    std::vector<std::vector<float>> img(H, std::vector<float>(W));
    std::vector<std::vector<float>> img_noise(H, std::vector<float>(W));
    
    // Execution time measurement
    auto start =  std::chrono::high_resolution_clock::now();

    // Do IDFT based on DFT results from task1_1
    IDFT_2D(dft_lena, img, H, W);
    IDFT_2D(dft_lena_noise, img_noise, H, W);

    std::vector<uint8_t> out1(H*W), out2(H*W);

    // Convert to uint8_t with clamping
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {

            // clamp to [0,255]
            float v1 = std::clamp(img[i][j], 0.0f, 255.0f);
            float v2 = std::clamp(img_noise[i][j], 0.0f, 255.0f);

            // convert to uint8_t and store
            out1[i * W + j] = (uint8_t)v1;
            out2[i * W + j] = (uint8_t)v2;
        }

    // save RAW
    FILE* f1_out = fopen(out_restore, "wb");
    FILE* f2_out = fopen(out_restore_noise, "wb");

    fwrite(out1.data(), 1, H*W, f1_out);
    fwrite(out2.data(), 1, H*W, f2_out);

    fclose(f1_out);
    fclose(f2_out);

    // save as png
    cv::Mat img_out(H, W, CV_8UC1, out1.data());
    cv::Mat img_out_noise(H, W, CV_8UC1, out2.data());

    cv::imwrite(std::string(out_restore) + ".png", img_out);
    cv::imwrite(std::string(out_restore_noise) + ".png", img_out_noise);

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "task1_2 done: IDFT restored images saved.\n";
    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Custom IDFT time: " << duration << " ms\n";

    // Compare the restored images with original images
    // Do PSNR to lena256.raw and restored lena
    FILE* lena_original = fopen("lena256.raw", "rb");
    FILE* lena_noise_original = fopen("lena256_noise.raw", "rb");

    std::vector<uint8_t> original_lena(H * W);
    std::vector<uint8_t> original_noise_lena(H * W);

    fread(original_lena.data(), 1, H * W, lena_original);
    fread(original_noise_lena.data(), 1, H * W, lena_noise_original);

    fclose(lena_noise_original);
    fclose(lena_original);

    double psnr_value_lena = computePSNR(original_lena, out1);
    double psnr_value_noise = computePSNR(original_noise_lena, out2);

    std::cout << "PSNR between original and IDFT-restored image = " << psnr_value_lena << " dB\n";
    std::cout << "PSNR between original noisy and IDFT-restored noisy image = " << psnr_value_noise << " dB\n";
}

// Implement built-in OpenCV DFT and IDFT for comparison
static void task1_3(const char* lena_raw, const char* lena_noise_raw,
                    const char* lena_mag_out, const char* lena_noise_mag_out,
                    const char* lena_ifft_out, const char* lena_noise_ifft_out)
{
    const int H = 256, W = 256;

    std::vector<uint8_t> input_lena_buffer(H * W), input_lena_noise_buffer(H * W);

    FILE* lena_original = fopen(lena_raw, "rb");
    FILE* lena_noise = fopen(lena_noise_raw, "rb");

    if (!lena_original || !lena_noise) {
        std::cerr << "Error: cannot open raw files.\n";
        return;
    }

    fread(input_lena_buffer.data(), 1, H * W, lena_original);
    fread(input_lena_noise_buffer.data(), 1, H * W, lena_noise);

    fclose(lena_original);
    fclose(lena_noise);

    // Execution time measurement
    auto start_dft = std::chrono::high_resolution_clock::now();

    cv::Mat img(H, W, CV_8UC1, input_lena_buffer.data());
    cv::Mat img_noise(H, W, CV_8UC1, input_lena_noise_buffer.data());

    // convert to float, CV_32F: 32-bit float single channel
    cv::Mat img_float, img_noise_float;

    img.convertTo(img_float, CV_32F);
    img_noise.convertTo(img_noise_float, CV_32F);

    cv::Mat shift(H, W, CV_32F), shift_noise(H, W, CV_32F);

    // Apply origin shift (-1)^(x+y)
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {

            // (-1)^(x+y)
            float s = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            shift.at<float>(i, j) = img_float.at<float>(i, j) * s;
            shift_noise.at<float>(i, j) = img_noise_float.at<float>(i, j) * s;
        }

    // Do DFT
    /*
        cv::dft(src, dst, flags), cv::DFT_COMPLEX_OUTPUT: get complex output
    */
    cv::Mat fft_img, fft_img_noise;
    cv::dft(shift, fft_img, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(shift_noise, fft_img_noise, cv::DFT_COMPLEX_OUTPUT);

    // cv::split to get real and imaginary parts
    // cv::split(src, mv), mv: array of Mat to hold splitted planes
    cv::Mat planes[2], planes2[2];
    cv::split(fft_img, planes);
    cv::split(fft_img_noise, planes2);


    // Compute magnitude spectrum
    cv::Mat mag, mag_noise;
    cv::magnitude(planes[0], planes[1], mag);
    cv::magnitude(planes2[0], planes2[1], mag_noise);

    // Log scale for better visualization
    mag += 1;
    mag_noise += 1;

    // cv::log(src, dst)
    // cv::log: computes the natural logarithm of each element in the source matrix.
    cv::log(mag, mag);
    cv::log(mag_noise, mag_noise);

    // Normalize to [0,255]
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    cv::normalize(mag_noise, mag_noise, 0, 255, cv::NORM_MINMAX);

    // Convert to uint8
    mag.convertTo(mag, CV_8U);
    mag_noise.convertTo(mag_noise, CV_8U);

    cv::imwrite(std::string(lena_mag_out) + ".png", mag);
    cv::imwrite(std::string(lena_noise_mag_out) + ".png", mag_noise);

    // Ends time measurement
    auto end_dft = std::chrono::high_resolution_clock::now();

    // DO IDFT
    /*
        cv::dft(src, dst, flags)
        cv::DFT_INVERSE: perform inverse DFT
        cv::DFT_REAL_OUTPUT: output is real numbers only
        cv::DFT_SCALE: scale the result by 1/(M*N)
    */

    // Start time measurement(IDFT)
    auto start_ifft = std::chrono::high_resolution_clock::now();

    cv::Mat ifft_img, ifft_img_noise;
    cv::dft(fft_img, ifft_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cv::dft(fft_img_noise, ifft_img_noise, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Undo origin shift
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            
            // (-1)^(x+y)
            float s = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            ifft_img.at<float>(i, j) *= s;
            ifft_img_noise.at<float>(i, j) *= s;
        }

    // Convert to uint8
    cv::Mat ifft_u8, ifft_noise_u8;
    ifft_img.convertTo(ifft_u8, CV_8U);
    ifft_img_noise.convertTo(ifft_noise_u8, CV_8U);

    // save raw
    FILE* fr1 = fopen(lena_ifft_out, "wb");
    FILE* fr2 = fopen(lena_noise_ifft_out, "wb");

    fwrite(ifft_u8.data, 1, H * W, fr1);
    fwrite(ifft_noise_u8.data, 1, H * W, fr2);

    fclose(fr1);
    fclose(fr2);

    // save as png
    cv::imwrite(std::string(lena_ifft_out) + ".png", ifft_u8);
    cv::imwrite(std::string(lena_noise_ifft_out) + ".png", ifft_noise_u8);

    // End time measurement
    auto end_ifft = std::chrono::high_resolution_clock::now();

    std::cout << "task1_3: OpenCV FFT + IFFT done.\n";

    // Calculate elapsed time
    auto duration_dft = std::chrono::duration_cast<std::chrono::milliseconds>(end_dft - start_dft).count();
    auto duration_ifft = std::chrono::duration_cast<std::chrono::milliseconds>(end_ifft - start_ifft).count();

    std::cout << "OpenCV DFT time: " << duration_dft << " ms\n";
    std::cout << "OpenCV IDFT time: " << duration_ifft << " ms\n";

    // PSNR comparison
    double psnr_value_lena = computePSNR(input_lena_buffer, std::vector<uint8_t>(ifft_u8.data, ifft_u8.data + H * W));
    double psnr_value_noise = computePSNR(input_lena_noise_buffer, std::vector<uint8_t>(ifft_noise_u8.data, ifft_noise_u8.data + H * W));
    std::cout << "PSNR between original and OpenCV IDFT-restored image = " << psnr_value_lena << " dB\n";
    std::cout << "PSNR between original noisy and OpenCV IDFT-restored noisy image = " << psnr_value_noise << " dB\n";
}

// DCT IDCT
static void task1_4(const char* input_lena, const char* input_lena_noise,const char* out_dct, const char* out_dct_noise, const char* out_idct, const char* out_idct_noise)
{
    const int H = 256, W = 256;

    std::vector<uint8_t> input_lena_buffer(H * W), input_lena_noise_buffer(H * W);
    FILE* lena_input = fopen(input_lena, "rb");
    FILE* lena_noise_input = fopen(input_lena_noise, "rb");

    fread(input_lena_buffer.data(), 1, H * W, lena_input);
    fread(input_lena_noise_buffer.data(), 1, H * W, lena_noise_input);

    fclose(lena_input);
    fclose(lena_noise_input);

    // Convert to cv::Mat
    cv::Mat img(H, W, CV_8UC1, input_lena_buffer.data());
    cv::Mat img_noise(H, W, CV_8UC1, input_lena_noise_buffer.data());

    cv::Mat img_float, img_noise_float;
    img.convertTo(img_float, CV_32F);
    img_noise.convertTo(img_noise_float, CV_32F);

    // DO DCT
    // https://en.wikipedia.org/wiki/Discrete_cosine_transform
    cv::Mat dct_img, dct_img_noise;
    cv::dct(img_float, dct_img);
    cv::dct(img_noise_float, dct_img_noise);

    // Magnitude (absolute values)
    cv::Mat mag = cv::abs(dct_img);
    cv::Mat mag_noise = cv::abs(dct_img_noise);

    // Add 1 to avoid log(0)
    mag += 1;
    mag_noise += 1;

    // Log scale to compress large dynamic range
    cv::log(mag, mag);
    cv::log(mag_noise, mag_noise);

    // gamma correction
    double gamma = 0.4;
    cv::pow(mag, gamma, mag);
    cv::pow(mag_noise, gamma, mag_noise);

    // Normalize to [0, 255]
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    cv::normalize(mag_noise, mag_noise, 0, 255, cv::NORM_MINMAX);

    // Convert to uint8
    mag.convertTo(mag, CV_8U);
    mag_noise.convertTo(mag_noise, CV_8U);

    // Save DCT results
    FILE* dct1_file = fopen(out_dct, "wb");
    FILE* dct2_file = fopen(out_dct_noise, "wb");

    fwrite(dct_img.data, sizeof(float), H * W, dct1_file);
    fwrite(dct_img_noise.data, sizeof(float), H * W, dct2_file);

    fclose(dct1_file);
    fclose(dct2_file);

    // save as png
    cv::imwrite(std::string(out_dct) + ".png", mag);
    cv::imwrite(std::string(out_dct_noise) + ".png", mag_noise);

    // DO IDCT
    cv::Mat idct_img, idct_img_noise;
    cv::idct(dct_img, idct_img);
    cv::idct(dct_img_noise, idct_img_noise);

    // Convert to uint8
    cv::Mat idct_u8, idct_noise_u8;
    idct_img.convertTo(idct_u8, CV_8U);
    idct_img_noise.convertTo(idct_noise_u8, CV_8U);

    // save raw
    FILE* idct1_file = fopen(out_idct, "wb");
    FILE* idct2_file = fopen(out_idct_noise, "wb");

    fwrite(idct_u8.data, 1, H * W, idct1_file);
    fwrite(idct_noise_u8.data, 1, H * W, idct2_file);

    fclose(idct1_file);
    fclose(idct2_file);

    // save as png
    cv::imwrite(std::string(out_idct) + ".png", idct_u8);
    cv::imwrite(std::string(out_idct_noise) + ".png", idct_noise_u8);

    std::cout << "task1_4: DCT + IDCT done.\n";
}

static void task1_5() {
    std::cout << "Please view report\n";
}

static void task2(const char* input_building, const char* output_n1, const char* output_n5)  
{
    const int H = 960, W = 540;
    const float D0 = 80.0f;

    std::vector<uint8_t> building_buffer(H * W);
    FILE* f = fopen(input_building, "rb");

    if (!f) { std::cerr << "Cannot open building raw.\n"; return; }

    fread(building_buffer.data(), 1, H * W, f);
    fclose(f);

    cv::Mat img(H, W, CV_8UC1, building_buffer.data());
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F);

    // shift origin to center by multiplying (-1)^(x+y)
    cv::Mat shifted(H, W, CV_32F);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            shifted.at<float>(i, j) = img_float.at<float>(i, j) * (( (i+j)%2==0 ) ? 1 : -1);

    // DO DFT
    cv::Mat dft_img;
    cv::dft(shifted, dft_img, cv::DFT_COMPLEX_OUTPUT);

    // CV_32F: 32-bit float single channel
    cv::Mat D(H, W, CV_32F);

    int cx = H/2;
    int cy = W/2;
    for (int u = 0; u < H; u++)
        for (int v = 0; v < W; v++)
        {
            float du = u - cx;
            float dv = v - cy;

            // Euclidean distance D(u,v) from center
            D.at<float>(u,v) = std::sqrt(du*du + dv*dv);
        }

    // DO Butterworth Low-Pass Filter (n=1 and n=5)
    cv::Mat H1(H, W, CV_32F), H5(H, W, CV_32F);

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            float d = D.at<float>(i,j);

            H1.at<float>(i,j) = 1.0f / (1.0f + std::pow(d / D0, 2 * 1)); // n=1
            H5.at<float>(i,j) = 1.0f / (1.0f + std::pow(d / D0, 2 * 5)); // n=5
        }

    cv::Mat dft_n1, dft_n5;

    // give DFT and filter mask to apply filter

    cv::Mat planes_for_filter[2];
    cv::split(dft_img, planes_for_filter);

    Apply_butterworth_filter(dft_n1, H1, planes_for_filter, H, W);
    Apply_butterworth_filter(dft_n5, H5, planes_for_filter, H, W);

    // Save filtered magnitude spectrum for n=1 and n=5
    processAndSaveMagnitude(dft_n1, "building_dft_butterworth_n1_mag.png");
    processAndSaveMagnitude(dft_n5, "building_dft_butterworth_n5_mag.png");

    // Perform IDFT for n=1
    cv::Mat idft_n1, idft_n1_shifted;
    cv::dft(dft_n1, idft_n1, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    idft_n1_shifted = cv::Mat(H, W, CV_32F);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            idft_n1_shifted.at<float>(i, j) = idft_n1.at<float>(i, j) * (( (i+j)%2==0 ) ? 1 : -1);

    cv::Mat img_n1;
    cv::normalize(idft_n1_shifted, idft_n1_shifted, 0, 255, cv::NORM_MINMAX);
    idft_n1_shifted.convertTo(img_n1, CV_8U);

    // save png for n=1
    cv::imwrite("building_idft_butterworth_n1.png", img_n1);

    // Perform IDFT for n=5
    cv::Mat idft_n5, idft_n5_shifted;
    cv::dft(dft_n5, idft_n5, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    idft_n5_shifted = cv::Mat(H, W, CV_32F);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            idft_n5_shifted.at<float>(i, j) = idft_n5.at<float>(i, j) * (( (i+j)%2==0 ) ? 1 : -1);

    cv::Mat img_n5;
    cv::normalize(idft_n5_shifted, idft_n5_shifted, 0, 255, cv::NORM_MINMAX);
    idft_n5_shifted.convertTo(img_n5, CV_8U);

    // save raw
    FILE* f_n1 = fopen(output_n1, "wb");
    FILE* f_n5 = fopen(output_n5, "wb");

    fwrite(img_n1.data, 1, H * W, f_n1);
    fwrite(img_n5.data, 1, H * W, f_n5);

    fclose(f_n1);
    fclose(f_n5);

    // save as png
    cv::imwrite(std::string(output_n1) + ".png", img_n1);
    cv::imwrite(std::string(output_n5) + ".png", img_n5);

    std::cout << "task2 done: Butterworth LPF (n=1,n=5) applied.\n";
}

static void task3_1(const char* lena_raw, const char* watermark_raw, const char* output_watermarked)
{
    /*
        divide lena256.raw into 8x8 blocks denoted by f(i,j)
    */
    const int H = 256, W = 256;
    const int Watermark_H = 32, Watermark_W = 32;

    std::vector<uint8_t> lena_buffer(H * W);
    FILE* lena_input = fopen(lena_raw, "rb");
    fread(lena_buffer.data(), 1, H * W, lena_input);
    fclose(lena_input);

    // build Mat
    cv::Mat lena(H, W, CV_8UC1, lena_buffer.data());
    lena.convertTo(lena, CV_32F);

    std::vector<uint8_t> watermark_buffer(Watermark_H * Watermark_W);
    FILE* input_watermark = fopen(watermark_raw, "rb");
    fread(watermark_buffer.data(), 1, Watermark_H * Watermark_W, input_watermark);
    fclose(input_watermark);

    // Convert to bitstream wi (0/1)
    // Binarize watermark: threshold at 127
    std::vector<int> watermark_bits(Watermark_H * Watermark_W);
    for (int i = 0; i < Watermark_H * Watermark_W; i++)
        watermark_bits[i] = (watermark_buffer[i] > 127 ? 1 : 0);

    cv::Mat watermarked = cv::Mat::zeros(H, W, CV_32F);
    int bit_index = 0;

    for (int by = 0; by < H; by += 8)
    {
        for (int bx = 0; bx < W; bx += 8)
        {
            // Extract 8×8 block
            // cv::Rect(x, y, width, height) => define rectangle
            /*
            For every block Ci:
            I. You may want to convert the block's pixel values to float format.
            II. Apply DCT
            III. From this matrix,extract two mid-frequency coefficients:
                A = C(2,3)
                B = C(3,2)

            The core embedding logic is then applied:

            * If the watermark bit wi =1 and A < B,
                then enforce A > B by setting:
                    C(2,3) = B + 10
            * If the watermark bit wi =0 and A >= B,
                then enforce A < B by setting:
                    C(3,2) = A + 10
            IV. Apply IDCT to get the modified block.
            */
            cv::Mat block = lena(cv::Rect(bx, by, 8, 8));
            cv::Mat block_float;
            block.convertTo(block_float, CV_32F);

            // DO DCT
            cv::Mat block_dct;
            cv::dct(block_float, block_dct);

            // Extract A and B
            float A = block_dct.at<float>(2, 3);
            float B = block_dct.at<float>(3, 2);

            int wi = watermark_bits[bit_index++];

            if (wi == 1 && A < B)
                block_dct.at<float>(2, 3) = B + 10;

            else if (wi == 0 && A >= B)
                block_dct.at<float>(3, 2) = A + 10;

            cv::Mat block_idct;
            cv::idct(block_dct, block_idct);

            // Copy back
            block_idct.copyTo(watermarked(cv::Rect(bx, by, 8, 8)));
        }
    }

    // Convert back to uint8
    cv::Mat final_img;
    watermarked.convertTo(final_img, CV_8U);

    FILE* fout = fopen(output_watermarked, "wb");
    fwrite(final_img.data, 1, H * W, fout);
    fclose(fout);

    // save as png
    cv::imwrite(std::string(output_watermarked) + ".png", final_img);

    // PSNR calculation
    double psnr_value = computePSNR(lena_buffer, std::vector<uint8_t>(final_img.data, final_img.data + H * W));
    std::cout << "PSNR between original Lena and watermarked Lena = " << psnr_value << " dB\n";

    std::cout << "task3_1 done: Watermark embedded into Lena.\n";
}

static void task3_2(const char* watermarked_raw, const char* blurred_raw, const char* blur_output, const char* watermark_recover_output)
{
    const int H = 256, W = 256;
    const int WM_H = 32, WM_W = 32;

    std::vector<uint8_t> watermark_buffer(H * W);
    FILE* watermark_input = fopen(watermarked_raw, "rb");
    fread(watermark_buffer.data(), 1, H*W, watermark_input);
    fclose(watermark_input);

    cv::Mat watermark_img(H, W, CV_8UC1, watermark_buffer.data());
    watermark_img.convertTo(watermark_img, CV_32F);

    cv::Mat blurred;
    gaussianBlur3x3(watermark_img, blurred);

    // Save blurred image
    cv::Mat blurred_u8;
    blurred.convertTo(blurred_u8, CV_8U);

    FILE* f2 = fopen(blur_output, "wb");
    fwrite(blurred_u8.data, 1, H*W, f2);
    fclose(f2);

    // save as png
    cv::imwrite(std::string(blur_output) + ".png", blurred_u8);

    std::vector<uint8_t> watermark_recover(WM_H * WM_W);
    int bit_idx = 0;

    for (int by = 0; by < H; by += 8)
    {
        for (int bx = 0; bx < W; bx += 8)
        {
            cv::Mat block = blurred(cv::Rect(bx, by, 8, 8));
            cv::Mat block_float;
            block.convertTo(block_float, CV_32F);

            // DCT
            cv::Mat block_dct;
            cv::dct(block_float, block_dct);

            float A = block_dct.at<float>(2,3);
            float B = block_dct.at<float>(3,2);

            // Extraction rule from task3_1
            /*
            if (A > B) bit = 1;
            else       bit = 0;
            */
            watermark_recover[bit_idx++] = (A > B ? 1 : 0);
        }
    }

    // Convert to image format (0/255)
    std::vector<uint8_t> watermark_img_out(WM_H * WM_W);
    for (int i = 0; i < WM_H * WM_W; i++)
        watermark_img_out[i] = (watermark_recover[i] ? 255 : 0);

    FILE* f3 = fopen(watermark_recover_output, "wb");
    fwrite(watermark_img_out.data(), 1, WM_H * WM_W, f3);
    fclose(f3);

    // save as png for easy viewing
    cv::Mat watermark_png(WM_H, WM_W, CV_8UC1, watermark_img_out.data());
    cv::imwrite(std::string(watermark_recover_output) + ".png", watermark_png);

    // PSNR calculation
    std::vector<uint8_t> original_watermark_buffer(WM_H * WM_W);
    FILE* original_wm_input = fopen("watermark32x32.raw", "rb");
    fread(original_watermark_buffer.data(), 1, WM_H * WM_W, original_wm_input);
    fclose(original_wm_input);

    double psnr_value = computePSNR(original_watermark_buffer, watermark_img_out);
    std::cout << "PSNR between original watermark and recovered watermark = " << psnr_value << " dB\n";

    std::cout << "task3_2 done: blurred image + recovered watermark saved.\n";
}

// helpers for task3_3

// Embed watermark subroutine
cv::Mat embedWatermark(cv::Mat& source_image, const std::vector<int>& watermark_bits, cv::Point A_pos, cv::Point B_pos)
{
    cv::Mat dst = cv::Mat::zeros(source_image.rows, source_image.cols, CV_32F);
    int bit_idx = 0;

    for(int by = 0; by < source_image.rows; by += 8)
    {
        for(int bx = 0; bx < source_image.cols; bx += 8)
        {
            // Follow the same embedding procedure as in task3_1
            // create 8x8 block
            cv::Mat block = source_image(cv::Rect(bx, by, 8, 8));
            cv::Mat fblock;

            // convert to float
            block.convertTo(fblock, CV_32F);

            cv::Mat C;
            // DO DCT
            cv::dct(fblock, C);

            // Extract A and B
            float& A = C.at<float>(A_pos.y, A_pos.x);
            float& B = C.at<float>(B_pos.y, B_pos.x);

            // Get watermark bit
            int watermark_index = watermark_bits[bit_idx++];

            // Embedding logic
            if (watermark_index == 1 && A < B)  A = B + 10;
            else if (watermark_index == 0 && A >= B)  B = A + 10;

            // DO IDCT for modified block
            cv::Mat idct_block;
            cv::idct(C, idct_block);

            idct_block.copyTo(dst(cv::Rect(bx, by, 8, 8)));
        }
    }
    return dst;
}

// Extract watermark subroutine
std::vector<uint8_t> extractWatermark(cv::Mat& source_image, cv::Point A_pos, cv::Point B_pos)
{
    // Each bit is determined by comparing the two DCT coefficients A and B
    std::vector<uint8_t> recover((source_image.rows / 8) * (source_image.cols / 8));
    int idx = 0;

    for (int by = 0; by < source_image.rows; by += 8)
    {
        for (int bx = 0; bx < source_image.cols; bx += 8)
        {
            // Create 8x8 block
            cv::Mat block = source_image(cv::Rect(bx, by, 8, 8));

            // Convert to float
            cv::Mat float_block;
            block.convertTo(float_block, CV_32F);

            cv::Mat C;
            // DO DCT
            cv::dct(float_block, C);

            // Extract A and B
            float A = C.at<float>(A_pos.y, A_pos.x);
            float B = C.at<float>(B_pos.y, B_pos.x);

            // Extraction rule(if A > B, bit = 1; else bit = 0)(255=white, 0=black)
            recover[idx++] = (A > B ? 255 : 0);
        }
    }
    return recover;
}

static void task3_3(const char* lena_raw, const char* watermark_raw,
                    const char* out_low_freq_watermark, const char* out_low_freq_blur,
                    const char* out_low_freq_recover, const char* out_high_freq_watermark,
                    const char* out_high_freq_blur, const char* out_high_freq_recover)
{
    const int H = 256, W = 256;
    const int WM_H = 32, WM_W = 32;

    std::vector<uint8_t> lena_buffer(H * W);
    FILE* lena_input = fopen(lena_raw, "rb");
    fread(lena_buffer.data(), 1, H * W, lena_input);
    fclose(lena_input);

    cv::Mat lena(H, W, CV_8UC1, lena_buffer.data());
    lena.convertTo(lena, CV_32F);

    std::vector<uint8_t> watermark_buffer(WM_H * WM_W);
    FILE* watermark_input = fopen(watermark_raw, "rb");
    fread(watermark_buffer.data(), 1, WM_H * WM_W, watermark_input);
    fclose(watermark_input);

    // Binarization
    std::vector<int> watermark_bits(WM_H * WM_W);
    for (int i = 0; i < WM_H * WM_W; ++i)
        watermark_bits[i] = (watermark_buffer[i] > 127 ? 1 : 0);

    // Case 1: Low freq (C[1,2], C[2,1])
    cv::Mat watermark_low_freq = embedWatermark(lena, watermark_bits, cv::Point(2, 1), cv::Point(1, 2));

    // Case 2: High freq (C[6,7], C[7,6])
    cv::Mat watermark_high_freq = embedWatermark(lena, watermark_bits, cv::Point(7, 6), cv::Point(6, 7));

    // DO Gaussian Blur on both watermarked images
    cv::Mat blur_low, blur_high;
    gaussianBlur3x3(watermark_low_freq, blur_low);
    gaussianBlur3x3(watermark_high_freq, blur_high);

    // Extract watermarks from both blurred images
    std::vector<uint8_t> recover_low_freq  = extractWatermark(blur_low,  cv::Point(2, 1), cv::Point(1, 2));
    std::vector<uint8_t> recover_high_freq = extractWatermark(blur_high, cv::Point(7, 6), cv::Point(6, 7));

    cv::Mat u8_low_freq, u8_high_freq, u8_blur_low_freq, u8_blur_high_freq;
    watermark_low_freq.convertTo(u8_low_freq, CV_8U);
    watermark_high_freq.convertTo(u8_high_freq, CV_8U);
    blur_low.convertTo(u8_blur_low_freq, CV_8U);
    blur_high.convertTo(u8_blur_high_freq, CV_8U);

    // Save results
    FILE* low_freq_watermark_out = fopen(out_low_freq_watermark, "wb"); 
    FILE* low_freq_blur_out = fopen(out_low_freq_blur, "wb"); 
    FILE* low_freq_recover_out = fopen(out_low_freq_recover, "wb"); 

    FILE* high_freq_watermark_out = fopen(out_high_freq_watermark, "wb");
    FILE* high_freq_blur_out = fopen(out_high_freq_blur, "wb");
    FILE* high_freq_recover_out = fopen(out_high_freq_recover, "wb");

    fwrite(u8_low_freq.data, 1, H * W, low_freq_watermark_out); fclose(low_freq_watermark_out);
    fwrite(u8_blur_low_freq.data, 1, H * W, low_freq_blur_out); fclose(low_freq_blur_out);
    fwrite(u8_high_freq.data, 1, H * W, high_freq_watermark_out); fclose(high_freq_watermark_out);
    fwrite(u8_blur_high_freq.data, 1, H * W, high_freq_blur_out); fclose(high_freq_blur_out);

    // recovered watermark images
    fwrite(recover_low_freq.data(), 1, WM_H * WM_W, low_freq_recover_out); fclose(low_freq_recover_out);
    fwrite(recover_high_freq.data(), 1, WM_H * WM_W, high_freq_recover_out); fclose(high_freq_recover_out);

    // save as png
    cv::imwrite(std::string(out_low_freq_watermark) + ".png", u8_low_freq);
    cv::imwrite(std::string(out_low_freq_blur) + ".png", u8_blur_low_freq);
    cv::imwrite(std::string(out_low_freq_recover) + ".png", cv::Mat(WM_H, WM_W, CV_8UC1, recover_low_freq.data()));

    // Save as PNG for high-frequency embedding
    cv::imwrite(std::string(out_high_freq_watermark) + ".png", u8_high_freq);
    cv::imwrite(std::string(out_high_freq_blur) + ".png", u8_blur_high_freq);
    cv::imwrite(std::string(out_high_freq_recover) + ".png", cv::Mat(WM_H, WM_W, CV_8UC1, recover_high_freq.data()));

    // PSNR calculations
    double psnr_low = computePSNR(watermark_buffer, recover_low_freq);
    double psnr_high = computePSNR(watermark_buffer, recover_high_freq);
    std::cout << "PSNR of recovered watermark (low-frequency embedding) = " << psnr_low << " dB\n";
    std::cout << "PSNR of recovered watermark (high-frequency embedding) = " << psnr_high << " dB\n";

    std::cout << "task3_3 done: compare low-frequency vs high-frequency watermark embedding.\n";
}

// helper function of task3_4
// Subroutine for embedding watermark with margin
cv::Mat embedWatermarkWithMargin(cv::Mat& source_image, const std::vector<int>& watermark_bits, cv::Point A_pos, cv::Point B_pos, float delta)
{
    cv::Mat dst = cv::Mat::zeros(source_image.rows, source_image.cols, CV_32F);
    int bit_idx = 0;

    for (int by = 0; by < source_image.rows; by += 8)
    {
        for (int bx = 0; bx < source_image.cols; bx += 8)
        {
            // Create 8x8 block
            cv::Mat block = source_image(cv::Rect(bx, by, 8, 8));
            cv::Mat fblock;
            block.convertTo(fblock, CV_32F);

            cv::Mat C;
            cv::dct(fblock, C);

            // Extract A and B
            float& A = C.at<float>(A_pos.y, A_pos.x);
            float& B = C.at<float>(B_pos.y, B_pos.x);

            int wi = watermark_bits[bit_idx++];

            // Embedding logic 
            float diff = A - B;
            float m = 0.5f * (A + B);

            if (wi == 1)
            {
                if (diff < delta)
                {
                    // Ensure A - B >= delta
                    A = m + delta * 0.5f;
                    B = m - delta * 0.5f;
                }
            }
            else
            {
                if (diff > -delta)
                {
                    // Ensure B - A >= delta
                    A = m - delta * 0.5f;
                    B = m + delta * 0.5f;
                }
            }

            cv::Mat idct_block;
            cv::idct(C, idct_block);
            idct_block.copyTo(dst(cv::Rect(bx, by, 8, 8)));
        }
    }
    return dst;
}

// Bonus: improved watermarking with margin (Δ) between A and B
// to make the extracted bits more robust against distortions (e.g. blur).
static void task3_4(const char* lena_raw,
                    const char* watermark_raw,
                    const char* out_watermarked,
                    const char* out_blurred,
                    const char* out_watermark_recovered)
{
    const int H = 256, W = 256;
    const int WM_H = 32, WM_W = 32;
    const float DELTA = 30.0f;   // safety margin |A - B| >= DELTA

    std::vector<uint8_t> lena_buffer(H * W);
    FILE* lena_input = fopen(lena_raw, "rb");
    fread(lena_buffer.data(), 1, H * W, lena_input);
    fclose(lena_input);

    cv::Mat lena(H, W, CV_8UC1, lena_buffer.data());
    lena.convertTo(lena, CV_32F);

    // Binarize watermark
    std::vector<uint8_t> watermark_buffer(WM_H * WM_W);
    FILE* watermark_input = fopen(watermark_raw, "rb");
    fread(watermark_buffer.data(), 1, WM_H * WM_W, watermark_input);
    fclose(watermark_input);

    std::vector<int> watermark_bits(WM_H * WM_W);
    for (int i = 0; i < WM_H * WM_W; ++i)
        watermark_bits[i] = (watermark_buffer[i] > 127 ? 1 : 0);

    cv::Mat watermarked = embedWatermarkWithMargin(lena, watermark_bits, cv::Point(2, 3), cv::Point(3, 2), DELTA);

    cv::Mat watermark_u8;
    watermarked.convertTo(watermark_u8, CV_8U);

    FILE* watermarked_out = fopen(out_watermarked, "wb");
    fwrite(watermark_u8.data, 1, H * W, watermarked_out);
    fclose(watermarked_out);
    cv::imwrite(std::string(out_watermarked) + ".png", watermark_u8);

    cv::Mat blurred;
    gaussianBlur3x3(watermarked, blurred);

    cv::Mat blurred_u8;
    blurred.convertTo(blurred_u8, CV_8U);
    FILE* blurred_out = fopen(out_blurred, "wb");
    fwrite(blurred_u8.data, 1, H * W, blurred_out);
    fclose(blurred_out);
    cv::imwrite(std::string(out_blurred) + ".png", blurred_u8);

    std::vector<uint8_t> watermark_recover_bits = extractWatermark(blurred, cv::Point(2, 3), cv::Point(3, 2));

    FILE* watermark_recovered_out = fopen(out_watermark_recovered, "wb");
    fwrite(watermark_recover_bits.data(), 1, WM_H * WM_W, watermark_recovered_out);
    fclose(watermark_recovered_out);

    cv::Mat watermark_recover_img(WM_H, WM_W, CV_8UC1, watermark_recover_bits.data());
    cv::imwrite(std::string(out_watermark_recovered) + ".png", watermark_recover_img);

    // (optional) PSNR between original Lena and watermarked Lena
    double psnr_value = computePSNR(
        lena_buffer,
        std::vector<uint8_t>(watermark_u8.data, watermark_u8.data + H * W)
    );
    std::cout << "Task3_4: PSNR = " << psnr_value << " dB\n";
    std::cout << "task3_4 done: improved watermarking with margin.\n";
}

int main() {
    std::cout << "----- Homework 5 Menu -----\n";
    
    // Variable for task1_1 and task1_2
    std::vector<std::vector<std::complex<float>>> dft_lena, dft_lena_noise;
    
    while (true) {
        std::cout << "\n================ Results Menu ================\n"
                  << " 1) task1_1\n"     
                  << " 2) task1_2\n"
                  << " 3) task1_3\n"
                  << " 4) task1_4\n"
                  << " 5) task1_5\n"
                  << " 6) task2\n"
                  << " 7) task3_1\n"
                  << " 8) task3_2\n"
                  << " 9) task3_3\n"
                  << " 10)task3_4\n"
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
            case 1: task1_1("lena256.raw", "lena256_noise.raw", dft_lena, dft_lena_noise); break;
            case 2: task1_2(dft_lena, dft_lena_noise, "idft_lena_image.raw", "idft_lena_noise_image.raw"); break;
            case 3: task1_3("lena256.raw", "lena256_noise.raw", "lena256_mag_opencv", "lena256_noise_mag_opencv", "lena256_idft_opencv.raw", "lena256_noise_idft_opnecv.raw"); break;
            case 4: task1_4("lena256.raw", "lena256_noise.raw", "lena256_dct.raw", "lena256_noise_dct.raw", "lena256_idct.raw", "lena256_noise_idct.raw"); break;
            case 5: task1_5(); break;
            case 6: task2("building_540x960.raw", "task2_n1.png", "task2_n5.png"); break;
            case 7: task3_1("lena256.raw", "watermark32x32.raw", "lena256_watermarked.raw"); break;
            case 8: task3_2("lena256_watermarked.raw","lena_watermarked_blur.raw", "lena_watermarked_blur.raw", "watermark_recover.raw"); break;
            case 9: task3_3("lena256.raw", "watermark32x32.raw", "task3_3_lowfreq_watermarked.raw", "task3_3_lowfreq_blurred.raw", "task3_3_lowfreq_recovered.raw", "task3_3_highfreq_watermarked.raw", "task3_3_highfreq_blurred.raw", "task3_3_highfreq_recovered.raw"); break;
            case 10:task3_4("lena256.raw", "watermark32x32.raw", "task3_4_watermarked.raw", "task3_4_blurred.raw", "task3_4_recovered.raw"); break;


            default: std::cout << "Unknown selection. Try 0-8.\n"; break; 
        }
    }
    return 0;
}
