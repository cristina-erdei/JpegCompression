#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

bool isInside(Mat img, int i, int j, int iMin, int iMax, int jMin, int jMax){
    return i >= iMin && i < iMax && j >= jMin && j < jMax;
}

double alpha(int n){
    return n == 0 ? sqrt(1.0/8.0) : sqrt(2.0/8.0);
}

//function performs the discrete cosine transformation for one position
double DCT(Mat_<uchar> img, int u, int v, int iMin, int iMax, int jMin, int jMax){
    double exteriorSum = 0.0;
    int iSize = iMax - iMin;
    int jSize = jMax - jMin;
    for(int x = 0; x < iSize; x++){
        double interiorSum = 0.0;
        for(int y = 0; y < jSize; y++){
            double arg1 = (M_PI / 8.0) * (x + 1.0/2.0) * (double)u;
            double arg2 = (M_PI / 8.0) * (y + 1.0/2.0) * (double)v;

            interiorSum += img(x + iMin, y + jMin) * cos(arg1) * cos(arg2);
        }
        exteriorSum += interiorSum;
    }

    return alpha(u) * alpha(v) * exteriorSum;
}


Mat_<uchar> compressBlock(Mat_<uchar> img, int iMin, int iMax, int jMin, int jMax){
    Mat_<uchar> compressedBlock(img.rows, img.cols);

    //scaling to range -127 <-> 128
    for(int i = iMin; i < iMax; i++){
        for(int j = jMin; j < jMax; j++){
            compressedBlock(i,j) = img(i, j) - 128;
        }
    }

    //apply DCT
    int iSize = iMax - iMin;
    int jSize = jMax - jMin;
    Mat_<float> A(iSize, jSize);
    for(int i= iMin; i < iMax; i++){
        for(int j = jMin; j < jMax; j++){
            A(i,j) = DCT(compressedBlock, i - iMin, j - jMin, iMin, iMax, jMin, jMax);
        }
    }

    //luminance matrix


    return compressedBlock;
}

Mat_<uchar> computeByBlocks(const Mat_<uchar>& img){

    Mat_<uchar> compressedImg(img.rows, img.cols);

    for(int i = 0; i < img.rows; i += 8){
        for(int j = 0; j < img.cols; j += 8){
            compressedImg = compressBlock(compressedImg, i, min(i + 8, img.rows), j, min(j + 8, img.cols));
            }
        }
    return compressedImg;
}


void JPEG_Compression(Mat_<Vec3b> original){
    Mat_<Vec3b> converted(original.rows, original.cols);
    cvtColor(original, converted, COLOR_BGR2YCrCb);

    imshow("Y Cr Cb image", converted);
    waitKey(0);

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

    computeByBlocks(Y);
    computeByBlocks(Cr);
    computeByBlocks(Cb);

    //todo put them back together
}


int main() {
    Mat_<Vec3b> image = imread("/home/cristinamicula/Documents/Facultate/IP/project/images/Lena_24bits.bmp");

    imshow("original", image);
    waitKey(0);

    JPEG_Compression(image);
    return 0;
}
