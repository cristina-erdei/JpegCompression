#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

bool isInside(Mat img, int i, int j){
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

double alpha(int n){
    return n == 0 ? sqrt(1.0/8.0) : sqrt(2.0/8.0);
}

double DCT(Mat_<uchar> block, int u, int v){
    double exteriorSum = 0.0;
    for(int x = 0; x < 7; x++){
        double interiorSum = 0.0;
        for(int y = 0; y < 7; y++){
            double arg1 = (M_PI / 8.0) * (x + 1.0/2.0) * (double)u;
            double arg2 = (M_PI / 8.0) * (y + 1.0/2.0) * (double)v;

            interiorSum += block(x,y) * cos(arg1) * cos(arg2);
        }
        exteriorSum += interiorSum;
    }

    return alpha(u) * alpha(v) * exteriorSum;
}


Mat_<uchar> compressedBlock(Mat_<uchar> block){
    if(block.rows != 8 || block.cols != 8){
        cout<<"ERROR: block has incorrect sizes.";
        exit(1);
    }

    Mat_<uchar> compressedBlock(block.rows, block.cols);

    //scaling to range -127 <-> 128
    for(int i = 0; i < block.rows; i++){
        for(int j = 0; j < block.cols; j++){
            compressedBlock(i,j) = block(i,j) - 128;
        }
    }

    //TODO: should here be double instead of float?
    Mat_<float> A(block.rows, block.cols);
    //apply DCT
    for(int i= 0; i < block.rows; i++){
        for(int j = 0; j <block.cols; j++){
            A(i,j) = DCT(compressedBlock, i, j);
        }
    }

}

void computeBy8x8Blocks(Mat_<uchar> img){

    for(int i = 0; i < img.rows; i += 8){
        for(int j = 0; j < img.cols; j += 8){
            Mat_<uchar> block(8,8);
            for(int k = 0; k < 8; k++){
                for(int l = 0; l < 8; l++){
                    block(k,l) = img(i+k,j+l);
                }
            }
            //TODO why segmentation fault?
//            Mat_<uchar> compressed = compressedBlock(block);

            //add it back to image
        }
    }

    //return converted image
}

void JPEG_Compression(Mat_<Vec3b> original){
    Mat_<Vec3b> converted(original.rows, original.cols);
    cvtColor(original, converted, COLOR_BGR2YCrCb);

    Mat_<uchar> Y(original.rows, original.cols);
    Mat_<uchar> Cr(original.rows, original.cols);
    Mat_<uchar> Cb(original.rows, original.cols);

    for(int i = 0; i < original.rows; i++){
        for(int j = 0; j < original.cols; j++){
            Y(i,j) = original(i,j)[0];
            Cr(i,j) = original(i,j)[1];
            Cb(i,j) = original(i,j)[2];
        }
    }

    computeBy8x8Blocks(Y);
    computeBy8x8Blocks(Cr);
    computeBy8x8Blocks(Cb);


    imshow("converted", converted);
    waitKey(0);
}


int main() {
    Mat_<Vec3b> image = imread("/home/cristinamicula/Documents/Facultate/IP/project/images/Lena_24bits.bmp");

    imshow("original", image);
    waitKey(0);

    JPEG_Compression(image);
    return 0;
}
