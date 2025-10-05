/********************************************************
* Filename    : HW1_fixed.cpp
* Author      : Stanley Chueh
* Note        : ADIP HW1 - menu-driven (portable)
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

// ---------- Q1.2(a1) ----------
static void task_1_2_a_1(const unsigned char* original_image, int W, int H) {
    int target1 = 70 * W + 147;
    int ans = original_image[target1];
    std::cout << "Ans:1.2(a)(1) Pixel value at (70th,147th): " << ans << "\n";
}

// ---------- Q1.2(a2) ----------
static void task_1_2_a_2(const unsigned char* original_image, int W, int H) {
    int target = 17643; // (68,235) in 256x256 when linearized from 0
    int ans = original_image[target - 1];
    std::cout << "Ans:1.2(a)(2) Pixel value at (68th,235th): " << ans << "\n";
}

// ---------- Q1.2(b): copy raw ----------
static void task_1_2_b(const unsigned char* original_image, int W, int H, const char* outname) {
    FILE* output_file = std::fopen(outname, "wb");
    if (!output_file) {
        std::perror("fopen");
        std::cerr << "Failed to open file for writing: " << outname << "\n";
        return;
    }

    fwrite(original_image, 1, W * H, output_file);
    std::fclose(output_file);

    std::cout << "Ans:1.2(b) -> " << outname << " (saved)\n";
}

// ---------- helped used by 1.2(c) ----------

/*
	(x1,y1)-> top left corner of block1, (x2,y2)-> top left corner of block2
	SZ -> block size(64)
	Read block1(x1,y1) and block2(x2,y2), then swap them
*/
static void swap_blocks(uint8_t* img, int W, int x1, int y1, int x2, int y2, int SZ) {
    for (int yy = 0; yy < SZ; yy++) {
        for (int xx = 0; xx < SZ; xx++) {
            uint8_t& A = img[(y1 + yy) * W + (x1 + xx)]; //8-bit pixel uint8_t(save space)
            uint8_t& B = img[(y2 + yy) * W + (x2 + xx)];

			// swap A and B
            uint8_t t = A; 
			A = B; 
			B = t;
        }
    }
}

/*
	(x,y) -> top left corner of block
	SZ -> block size(64)
	Read block(x,y), then rotate it 90 degree clockwise
	EX:
	(0,0)->(0,64)
	(0,1)->(1,64)
	(0,2)->(2,64)....
*/
static void rotate90cw_block(uint8_t* img, int W, int x, int y, int SZ) {
    std::vector<uint8_t> tmp(SZ * SZ); //space for tmp block

	// copy out
    for (int r = 0; r < SZ; r++) 
		std::memcpy(&tmp[r * SZ], &img[(y + r) * W + x], SZ);

	// write back rotated: (r,c) -> (c, SZ-1-r)
    for (int r = 0; r < SZ; r++)
        for (int c = 0; c < SZ; c++)
            img[(y + c) * W + (x + (SZ - 1 - r))] = tmp[r * SZ + c];
}

/*
	(x,y) -> top left corner of block
	SZ -> block size(64)
	Read block(x,y), then rotate it 90 degree counter-clockwise
	EX:data
	(0,0)->(64,0)
	(0,1)->(63,0)
	(0,2)->(62,0)....
*/
static void rotate90ccw_block(uint8_t* img, int W, int x, int y, int SZ) {
    std::vector<uint8_t> tmp(SZ * SZ); //space for tmp block
	// copy out
    for (int r = 0; r < SZ; r++)
        for (int c = 0; c < SZ; c++)
            tmp[r * SZ + c] = img[(y + r) * W + (x + c)];
	// write back rotated: (r,c) -> (SZ-1-c, r)
    for (int r = 0; r < SZ; r++)
        for (int c = 0; c < SZ; c++)
            img[(y + (SZ - 1 - c)) * W + (x + r)] = tmp[r * SZ + c];
}

// ---------- Q1.2(c) ----------
static void task_1_2_c(const unsigned char* original_image, int W, int H, const char* outname) {
    const int TL = W / 2;    // 128
    const int SQ = 64;

    std::vector<uint8_t> trig_c1(TL * TL, 0), trig_c2(TL * TL, 0),
                         trig_c3(TL * TL, 0), trig_c4(TL * TL, 0);
    std::vector<uint8_t> sqr_c1(TL * TL, 0), sqr_c2(TL * TL, 0),
                         sqr_c3(TL * TL, 0), sqr_c4(TL * TL, 0),
                         sqr_c5(TL * TL, 0), sqr_c6(TL * TL, 0),
                         sqr_c7(TL * TL, 0), sqr_c8(TL * TL, 0);

	// Extract 12 pieces
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        uint8_t px = original_image[i * W + j];
		
        if (i < TL && j < TL) {
			// Top-left triangle
            if (i >= j) 
				trig_c1[i * TL + j] = px; // c1: TL below
			else 
				trig_c2[i * TL + j] = px; // c2: TL above
        } 
		else if (i >= TL && j >= TL) {
			// Bottom-right triangle
            int li = i - TL, lj = j - TL;
            if (li >= lj)
				trig_c3[li * TL + lj] = px;  // c3: BR below
			else 
				trig_c4[li * TL + lj] = px; // c4: BR above
        } 
		else if (i < TL && j >= TL) {
			// Top-right squares
            int jj = j - TL; 
			size_t idx = (size_t)i * TL + jj; 
			int br = i / SQ, bc = jj / SQ;
            if(br == 0 && bc == 0)
				sqr_c1[idx] = px;
            else if (br == 0 && bc == 1)
				sqr_c2[idx] = px;
            else if (br == 1 && bc == 0)
				sqr_c3[idx] = px;
            else                         
				sqr_c4[idx] = px;
        } 
		else {
            int ii = i - TL; size_t idx = (size_t)ii * TL + j; int br = ii / SQ, bc = j / SQ;
            if (br == 0 && bc == 0) // Top-left square
				sqr_c5[idx] = px;
            else if (br == 0 && bc == 1) // Top-right square
				sqr_c6[idx] = px;
            else if (br == 1 && bc == 0) // Bottom-left square
				sqr_c7[idx] = px;
            else                      // Bottom-right square
				sqr_c8[idx] = px;
        }
    }

    // Copy original_image into a vector
    std::vector<uint8_t> dst(original_image, original_image + W * H);

	// Triangle remap

	// c1 -> TL upper mirrored
    for (int i = 0; i < TL; i++)
        for (int j = 0; j <= i; j++)
            dst[(size_t)(TL - 1 - i) * W + (TL - 1 - j)] = trig_c1[i * TL + j];
	// c2 -> BR upper
    for (int i = 0; i < TL; i++)
        for (int j = i + 1; j < TL; j++)
            dst[(size_t)(TL + i) * W + (TL + j)] = trig_c2[i * TL + j];
	// c3 -> TL lower
    for (int li = 0; li < TL; li++)
        for (int lj = 0; lj <= li; lj++)
            dst[(size_t)li * W + lj] = trig_c3[li * TL + lj];
	// c4 -> BR lower mirrored
    for (int li = 0; li < TL; li++)
        for (int lj = li + 1; lj < TL; lj++) {
            int yi = (W - 1) - li;
            int xj = (W - 1) - lj;
            dst[(size_t)yi * W + xj] = trig_c4[li * TL + lj];
        }

	// Square remap + rotate
    int TLx = TL, TLy = 0;
    int tr_x00 = TLx + 0 * SQ, tr_y00 = TLy + 0 * SQ;
    int tr_x01 = TLx + 1 * SQ, tr_y01 = TLy + 0 * SQ;
    int tr_x10 = TLx + 0 * SQ, tr_y10 = TLy + 1 * SQ;
    int tr_x11 = TLx + 1 * SQ, tr_y11 = TLy + 1 * SQ;

    int bl_x00 = 0 + 0 * SQ, bl_y00 = TL + 0 * SQ;
    int bl_x01 = 0 + 1 * SQ, bl_y01 = TL + 0 * SQ;
    int bl_x10 = 0 + 0 * SQ, bl_y10 = TL + 1 * SQ;
    int bl_x11 = 0 + 1 * SQ, bl_y11 = TL + 1 * SQ;
	
	// swap
    swap_blocks(dst.data(), W, tr_x00, tr_y00, bl_x00, bl_y00, SQ);
    swap_blocks(dst.data(), W, tr_x01, tr_y01, bl_x01, bl_y01, SQ);
    swap_blocks(dst.data(), W, tr_x10, tr_y10, bl_x10, bl_y10, SQ);
    swap_blocks(dst.data(), W, tr_x11, tr_y11, bl_x11, bl_y11, SQ);

	// rotate TR
    rotate90cw_block (dst.data(), W, tr_x00, tr_y00, SQ);
    rotate90ccw_block(dst.data(), W, tr_x01, tr_y01, SQ);
    rotate90ccw_block(dst.data(), W, tr_x10, tr_y10, SQ);
    rotate90cw_block (dst.data(), W, tr_x11, tr_y11, SQ);

	// rotate BL
    rotate90cw_block (dst.data(), W, bl_x00, bl_y00, SQ);
    rotate90ccw_block(dst.data(), W, bl_x01, bl_y01, SQ);
    rotate90ccw_block(dst.data(), W, bl_x10, bl_y10, SQ);
    rotate90cw_block (dst.data(), W, bl_x11, bl_y11, SQ);

	// Write to file
    FILE* output_file = std::fopen(outname, "wb");
    if (!output_file) 
	{ 
		std::perror("fopen"); 
		return; 
	}
	fwrite(dst.data(), 1, (size_t)W * H, output_file);
    fclose(output_file);
    std::cout << "Ans:1.2(c) -> " << outname << " (saved)\n";
}

// ---------- Q1.3(a): brightness +50 ----------
static void task_1_3_a(const unsigned char* original_image, int W, int H, const char* outname) {
    std::vector<uint8_t> out(original_image, original_image + W * H);

	for (int i=0;i< W*H;i++)
	{
		if (out[i] >= 205) // cap at 255
		{
			out[i] = 255;
			continue;
		}
		out[i] += 50;
	}

    FILE* f = std::fopen(outname, "wb");
	fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    std::cout << "Ans:1.3(a) -> " << outname << " (saved)\n";
}

// ---------- Q1.3(b): add noise [-70,70] ----------
static void task_1_3_b(const unsigned char* original_image, int W, int H, const char* outname) {
    std::vector<uint8_t> out(original_image, original_image + (size_t)W * H);

	for (int i=0;i< W*H;i++)
	{
		int noise = (rand() % 141) - 70; // [-70,70]
		int new_val = (int)out[i] + noise;
		if (new_val < 0) 
			new_val = 0;
		if (new_val > 255) 
			new_val = 255;
		out[i] = (uint8_t)new_val;
	}

    FILE* f = std::fopen(outname, "wb");
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    std::cout << "Ans:1.3(b) -> " << outname << " (saved)\n";
}

// ---------- Q2(a)(b): OpenCV drawing ----------
// Mat is a matrix contraining pixel values, Scalr(255,255,255) is white background
// CV_8UC3 means 8-bit unsigned, 3 channels (BGR)
static void task_2_ab(const char* png_name) {
    cv::Mat img(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));

	// outer red oval
	// ellipse(image, center,size, angle, startAngle, endAngle, color, thickness)
	//Size(width, height)
    cv::ellipse(img, cv::Point(200, 200), cv::Size(100, 70), 0, 0, 360, cv::Scalar(0, 0, 255), -1);

	// inner yellow oval
    cv::ellipse(img, cv::Point(200, 200), cv::Size(70, 50), 0, 0, 360, cv::Scalar(0, 255, 255), -1);

	// white oval (flower center)
    cv::ellipse(img, cv::Point(200, 200), cv::Size(40, 30), 0, 0, 360, cv::Scalar(255, 255, 255), -1);

	// two black eyes
    cv::ellipse(img, cv::Point(190, 200), cv::Size(5, 15), 0, 0, 360, cv::Scalar(0, 0, 0), -1);	
    cv::ellipse(img, cv::Point(210, 200), cv::Size(5, 15), 0, 0, 360, cv::Scalar(0, 0, 0), -1);

	// green rectangle (stem)
	// rectangle(image, Point(x1,y1), Point(x2,y2), color, thickness)
    cv::rectangle(img, cv::Point(195, 270), cv::Point(205, 350), cv::Scalar(0, 200, 0), -1);

	// leafs
    cv::ellipse(img, cv::Point(170, 310), cv::Size(40, 20), 30, 0, 360, cv::Scalar(0, 200, 0), -1);
    cv::ellipse(img, cv::Point(230, 310), cv::Size(40, 20), -30, 0, 360, cv::Scalar(0, 200, 0), -1);

	// leaf veins
	// line(image, Point(x1,y1), Point(x2,y2), color, thickness)
    cv::line(img, cv::Point(190, 320), cv::Point(145, 290), cv::Scalar(0, 0, 0), 2);
    cv::line(img, cv::Point(210, 320), cv::Point(255, 290), cv::Scalar(0, 0, 0), 2);

	// studnet ID
    std::string student_id = "114318047";
    cv::putText(img, student_id, cv::Point(130, 100), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);
	cv::imwrite(png_name, img);
    std::cout << "Ans:2(a)&(b) -> " << png_name << " (saved)\n";
}

int main() {
    const int W = 256, H = 256;
    const char* input_image  = "lena256.raw";
    const char* output_image = "lena256out.raw";

    // read original image
    unsigned char image_original[W * H];
    FILE* input_file = std::fopen(input_image, "rb");
    if (!input_file) {
        std::perror("fopen");
        std::cerr << "Cannot open file: " << input_image << "\n";
        return 1;
    }

    fread(image_original, 1, W * H, input_file);
    fclose(input_file);

    std::cout << "----- Homework 1 Menu -----\n";
    while (true) {
        std::cout << "\n================ Results Menu ================\n"
                  << " 1) 1.1-a&b (print note only)\n"
                  << " 2) 1.2a-1  (two pixel values)\n"
                  << " 3) 1.2a-2  (two pixel values)\n"
                  << " 4) 1.2b    (dump RAW copy)\n"
                  << " 5) 1.2c    (12-piece rearrange)\n"
                  << " 6) 1.3a    (brightness +50)\n"
                  << " 7) 1.3b    (add noise [-70,70])\n"
                  << " 8) 1.3c    (print note only)\n"
                  << " 9) 2.a&b   (draw PNG with OpenCV)\n"
                  << " 0) Exit\n"
                  << "Enter the question number: ";

        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(1 << 20, '\n');
            std::cout << "Invalid input. Please enter a number between 0 and 9.\n";
            continue;
        }
        if (choice == 0) break;

        switch (choice) {
            case 1: std::cout << "Ans:1.1(a)&(b) Please view report for details.\n"; break;
            case 2: task_1_2_a_1(image_original, W, H); break;
            case 3: task_1_2_a_2(image_original, W, H); break;
            case 4: task_1_2_b  (image_original, W, H, output_image); break; 
            case 5: task_1_2_c  (image_original, W, H, "lena256_1_2c_final.raw"); break;
            case 6: task_1_3_a  (image_original, W, H, "lena256_1_3a.raw"); break;
            case 7: task_1_3_b  (image_original, W, H, "lena256_1_3b.raw"); break;
            case 8: std::cout << "Ans:1.3(c) please view report for details.\n"; break;
            case 9: task_2_ab("fire_flower.png"); break;
            default: std::cout << "Unknown selection. Try 0-9.\n"; break;
        }
    }
    return 0;
}
