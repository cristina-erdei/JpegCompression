#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <utility>

using namespace cv;
using namespace std;

double alpha(int i, int N) {
    return i == 0 ? sqrt(1.0 / N) : sqrt(2.0 / N);
}

//function performs the discrete cosine transformation for one position in an 8x8 matrix
double DCT(Mat_<int> img, int i, int j) {
    double exteriorSum = 0.0;
    for (int x = 0; x < 8; x++) {
        double interiorSum = 0.0;
        for (int y = 0; y < 8; y++) {

            double arg1 = ((2 * x + 1) * i * M_PI) / (double) (2 * 8);
            double arg2 = ((2 * y + 1) * j * M_PI) / (double) (2 * 8);

            interiorSum += img(x, y) * cos(arg1) * cos(arg2);
        }
        exteriorSum += interiorSum;
    }

    return alpha(i, 8) * alpha(j, 8) * exteriorSum;
}

vector<int> runLengthEncoding(vector<int> zigzag) {
    vector<int> rle;

    for (int i = 0; i < zigzag.size(); i++) {
        int count = 1;
        int j = i + 1;
        while (j < zigzag.size() && zigzag[j] == zigzag[i]) {
            j++;
            count++;
        }
        rle.push_back(count);
        rle.push_back(zigzag[i]);
        i = j - 1;
    }
    return rle;
}

//assumes block is 8x8 matrix
vector<int> zigzag(Mat_<int> block) {
    vector<int> zigzag;
    zigzag.push_back(block(0, 0));
    int i = 0;
    int j = 1;

    //dir = 0: down-up diagonal
    //dir 1 : up-down diagonal
    int dir = 1;
    int count = 1;
    int upperHalfCount = block.rows * (block.rows + 1) / 2;
    int lowerHalfCount = block.rows * (block.rows - 1) / 2;
    int lowerCount = 0;

    //top left half of the block
    while (count < upperHalfCount) {
        zigzag.push_back(block(i, j));
        count++;
        if (dir == 1) {
            if (j == 0) {
                dir = 0;
                i++;
            } else {
                i++;
                j--;
            }
        } else {
            if (i == 0) {
                dir = 1;
                j++;
            } else {
                i--;
                j++;
            }
        }
    }

    dir = 0;
    i = block.rows - 1;
    j = 1;
    count = 0;

    //bottom right half of block
    while (count < lowerHalfCount) {
        zigzag.push_back(block(i, j));
        count++;

        if (dir == 1) {
            if (i == block.rows - 1) {
                dir = 0;
                j++;
            } else {
                i++;
                j--;
            }
        } else {
            if (j == block.cols - 1) {
                dir = 1;
                i++;
            } else {
                i--;
                j++;
            }
        }
    }
    return zigzag;
}

vector<int> compressBlock(Mat_<uchar> img, int iMin, int iMax, int jMin, int jMax) {
    //scaling to range -127 <-> 128
    Mat_<int> compressedBlock(8, 8);
    for (int i = iMin; i < iMax; i++) {
        for (int j = jMin; j < jMax; j++) {
            compressedBlock(i - iMin, j - jMin) = img(i, j) - 128;
        }
    }

    //apply DCT
    Mat_<int> A(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A(i, j) = (int) round(DCT(compressedBlock, i, j));
        }
    }
    //luminance matrix
    int values[64] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
    };
    Mat_<int> Q(8, 8, values);

    Mat_<int> B(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            B(i, j) = (int) round(A(i, j) / (double) Q(i, j));
        }
    }

    vector<int> zz = zigzag(B);
    vector<int> encoding = runLengthEncoding(zz);
    return encoding;
}

//expects image that has rows and cols a multiple of 8
vector<int> computeByBlocks(const Mat_<uchar> &img, int *totalNumberOfBlocks) {
    Mat_<uchar> compressedImg(img.rows, img.cols);
    compressedImg = img.clone();
    vector<int> entireEncoding;
    *totalNumberOfBlocks = 0;

    for (int i = 0; i < img.rows; i += 8) {
        for (int j = 0; j < img.cols; j += 8) {
            vector<int> aux = compressBlock(compressedImg, i, i + 8, j, j + 8);
            for (int x : aux) {
                entireEncoding.push_back(x);
            }
            (*totalNumberOfBlocks)++;
        }
    }
    return entireEncoding;
}

Mat_<Vec3b> addPadding(Mat_<Vec3b> img) {
    int addI = (8 - (img.rows % 8)) % 8;
    int addJ = (8 - (img.cols % 8)) % 8;
    Mat_<Vec3b> padded(img.rows + addI, img.cols + addJ);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            padded(i, j) = img(i, j);
        }
    }

    for (int i = img.rows; i < img.rows + addI; i++) {
        for (int j = img.cols; j < img.cols + addJ; j++) {
            padded(i, j) = {0, 0, 0};
        }
    }

    return padded;
}

void writeBinaryFile(vector<int> dataToWrite, const String &filename) {
    ofstream file(filename, ios::out | ios::binary | ios::trunc);
    if (!file.is_open()) {
        cout << "ERROR: cannot open file " << filename << endl;
        return;
    }
    file.write((char *) &dataToWrite[0], dataToWrite.size() * sizeof(dataToWrite[0]));
    file.close();
}

vector<int> readBinaryFile(const String &filename) {
    ifstream file(filename, ios::in | ios::binary);
    if (!file.is_open()) {
        cout << "ERROR: cannot open file " << filename << endl;
        exit(1);
    }

    vector<int> data;
    int aux;
    file.read((char *) &aux, sizeof(int));
    while (!file.eof()) {
        data.push_back(aux);
        file.read((char *) &aux, sizeof(int));
    }

    file.close();
    return data;
}

void JPEG_Compression(Mat_<Vec3b> original, const String &binaryFileName) {
    Mat_<Vec3b> padded = addPadding(std::move(original));
    Mat_<Vec3b> converted(padded.rows, padded.cols);
    cvtColor(padded, converted, COLOR_BGR2YCrCb);

    imshow("padded Y Cr Cb image", converted);
    waitKey(0);

    Mat_<uchar> Y(padded.rows, padded.cols);
    Mat_<uchar> Cr(padded.rows, padded.cols);
    Mat_<uchar> Cb(padded.rows, padded.cols);
    for (int i = 0; i < padded.rows; i++) {
        for (int j = 0; j < padded.cols; j++) {
            Y(i, j) = converted(i, j)[0];
            Cr(i, j) = converted(i, j)[1];
            Cb(i, j) = converted(i, j)[2];
        }
    }

    vector<int> fileContent;
    fileContent.push_back(converted.rows);
    fileContent.push_back(converted.cols);

    int totalBlocks;
    vector<int> aux = computeByBlocks(Y, &totalBlocks);
    fileContent.push_back(totalBlocks);

    for (int i : aux) {
        fileContent.push_back(i);
    }

    aux = computeByBlocks(Cr, &totalBlocks);
    for (int i : aux) {
        fileContent.push_back(i);
    }

    aux = computeByBlocks(Cb, &totalBlocks);
    for (int i : aux) {
        fileContent.push_back(i);
    }
    writeBinaryFile(fileContent, binaryFileName);
}

double inverse_DCT(Mat_<int> block, int x, int y) {
    double exteriorSum = 0.0;

    for (int i = 0; i < 8; i++) {
        double interiorSum = 0.0;
        for (int j = 0; j < 8; j++) {
            double arg1 = ((2 * x + 1) * i * M_PI) / (double) (2 * 8);
            double arg2 = ((2 * y + 1) * j * M_PI) / (double) (2 * 8);
            interiorSum += alpha(i, 8) * alpha(j, 8) * block(i, j) * cos(arg1) * cos(arg2);
        }
        exteriorSum += interiorSum;
    }
    return exteriorSum;
}

Mat_<int> reverseZigzag(vector<int> data) {
    Mat_<int> block(8, 8);
    block.setTo(0);
    block(0, 0) = data[0];
    int i = 0;
    int j = 1;
    //dir = 0: down-up diagonal
    //dir 1 : up-down diagonal
    int dir = 1;
    int count = 1;
    int upperHalfCount = block.rows * (block.rows + 1) / 2;

    //top left half of the block
    while (count < upperHalfCount) {
        block(i, j) = data[count];
        count++;
        if (dir == 1) {
            if (j == 0) {
                dir = 0;
                i++;
            } else {
                i++;
                j--;
            }
        } else {
            if (i == 0) {
                dir = 1;
                j++;
            } else {
                j++;
                i--;
            }
        }
    }

    dir = 0;
    i = block.rows - 1;
    j = 1;

    //bottom right half of block
    while (count < data.size()) {
        block(i, j) = data[count];
        count++;

        if (dir == 1) {
            if (i == block.rows - 1) {
                dir = 0;
                j++;
            } else {
                i++;
                j--;
            }
        } else {
            if (j == block.cols - 1) {
                dir = 1;
                i++;
            } else {
                i--;
                j++;
            }
        }
    }
    return block;
}

Mat_<uchar> reconstructBlock(vector<int> data, int start, int end) {
    Mat_<int> block(8, 8);
    vector<int> aux;
    for (int i = start; i < end; i++) {
        aux.push_back(data[i]);
    }
    block = reverseZigzag(aux);

    //luminance matrix
    int values[64] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
    };
    Mat_<int> Q(8, 8, values);
    Mat_<int> B(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            B(i, j) = block(i, j) * Q(i, j);
        }
    }

    Mat_<int> A(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A(i, j) = (int) round(inverse_DCT(B, i, j));
        }
    }

    Mat_<uchar> final(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int temp = A(i, j) + 128;
            if (temp > 255) {
                final(i, j) = 255;
            } else if (temp < 0) {
                final(i, j) = 0;
            } else {
                final(i, j) = temp;
            }
        }
    }
    return final;
}

Mat_<uchar>
reconstructMatrix(const vector<int> &data, int rows, int cols, int nrOfBlocks) {
    Mat_<uchar> matrix(rows, cols);
    int i = 0;
    int j = 0;
    int index = 0;
    for (int number = 0; number < nrOfBlocks; number++) {
        Mat_<uchar> block = reconstructBlock(data, index, index + 64);
        index += 64;
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                matrix(i + x, j + y) = block(x, y);
            }
        }
        j += 8;
        if (j == matrix.cols) {
            j = 0;
            i += 8;
        }
    }
    return matrix;
}


//flag = true - decode rle + complete zig zag
//flag = false - decode partial zig zag
void JPEG_Decompression(const String &binaryFilename, String outputFilename) {
    vector<int> data = readBinaryFile(binaryFilename);
    int rows = data[0];
    int cols = data[1];
    int nrOfBlocks = data[2];
    vector<int> zz[3];

    int index = 3;
    //rle decoding
    for (int nr = 0; nr < 3; nr++) {
        for (int block = 0; block < nrOfBlocks; block++) {
            int sum = 0;
            while (sum < 64) {
                int count = data[index];
                sum += count;
                index++;
                for (int i = 0; i < count; i++) {
                    zz[nr].push_back(data[index]);
                }
                index++;
            }
        }
    }

    Mat_<uchar> Y = reconstructMatrix(zz[0], rows, cols, nrOfBlocks);
    Mat_<uchar> Cr = reconstructMatrix(zz[1], rows, cols, nrOfBlocks);
    Mat_<uchar> Cb = reconstructMatrix(zz[2], rows, cols, nrOfBlocks);

    Mat_<Vec3b> image(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image(i, j)[0] = Y(i, j);
            image(i, j)[1] = Cr(i, j);
            image(i, j)[2] = Cb(i, j);
        }
    }

    imshow("compressed Y Cr Cb image", image);
    waitKey(0);

    Mat_<Vec3b> final(rows, cols);
    cvtColor(image, final, COLOR_YCrCb2BGR);


    imshow("compressed final image", final);
    waitKey(0);

    imwrite(outputFilename, final);
}

double calculateCompressionRatio(const String &initialImage, const String &compressedImage) {
    ifstream originalFile(initialImage, ios::binary);
    ifstream compressedFile(compressedImage, ios::binary);
    originalFile.seekg(0, ios::end);
    long size = originalFile.tellg();
    originalFile.seekg(0, ios::beg);
    compressedFile.seekg(0, ios::end);
    long size2 = compressedFile.tellg();
    compressedFile.seekg(0, ios::beg);
    return (double) size / (double) size2;
}

int main(int argc, char *argv[]) {
    String inputFilename = argv[1];
    String outputFilename = argv[2];
    String binaryFilename = argv[3];

    Mat_<Vec3b> image = imread(inputFilename);

    imshow("original", image);
    waitKey(0);

    JPEG_Compression(image, binaryFilename);
    double ratio = calculateCompressionRatio(inputFilename, outputFilename);
    cout << " Compression ratio: " << ratio << endl;
    JPEG_Decompression(binaryFilename, outputFilename);

    return 0;

}
