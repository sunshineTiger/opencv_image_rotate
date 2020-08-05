#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;


int showImage(Mat &src) {
    //白色背景变成黑色
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            if (src.at<Vec3b>(row, col) == Vec3b(255, 255, 255)) {
                src.at<Vec3b>(row, col)[0] = 0;
                src.at<Vec3b>(row, col)[1] = 0;
                src.at<Vec3b>(row, col)[2] = 0;
            }
        }
    }
    imshow("black backgroung", src);

    Mat x_grad, y_grad;
    Mat out_image;
    Sobel(src, x_grad, CV_16S, 1, 0, 3);
    Sobel(src, y_grad, CV_16S, 0, 1, 3);
    convertScaleAbs(x_grad, x_grad);
    convertScaleAbs(y_grad, y_grad);
    imshow("x_grad", x_grad);
    imshow("y_grad", y_grad);
    addWeighted(x_grad, 0.5, y_grad, 0.5, 0, out_image);
    imshow("addWeightedxx", out_image);



    //转换成二值图
    Mat binary;

    cvtColor(out_image, out_image, CV_BGR2GRAY);

    threshold(out_image, binary, 40, 255, THRESH_BINARY | THRESH_OTSU);

    imshow("binary image", binary);

    //距离变换

    Mat distImg;

    distanceTransform(binary, distImg, DIST_L1, 3, 5);

    normalize(distImg, distImg, 0, 1, NORM_MINMAX);

    imshow("dist image", distImg);

    //二值化

    threshold(distImg, distImg, 0.4, 1, THRESH_BINARY);

    imshow("dist binary image", distImg);

    //腐蚀(使得连在一起的部分分开)

    Mat k1 = Mat::ones(3, 3, CV_8UC1);

    erode(distImg, distImg, k1);

    imshow("分开", distImg);

    //标记

    Mat dist_8u;

    distImg.convertTo(dist_8u, CV_8U);

    vector<vector<Point>> contours;

    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    //创建标记

    Mat marker = Mat::zeros(src.size(), CV_32SC1);

    //画标记
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(marker, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }

    circle(marker, Point(5, 5), 3, Scalar(255, 255, 255), -1);
    imshow("marker", marker * 1000);

    //分水岭变换
    watershed(src, marker);//根据距离变换的标记，在原图上分离
    Mat water = Mat::zeros(marker.size(), CV_8UC1);
    marker.convertTo(water, CV_8UC1);
    bitwise_not(water, water, Mat());//取反操作
    //imshow("源 image", src);
    imshow("watershed Image", water);

    // generate random color
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int r = theRNG().uniform(100, 255);
        int g = theRNG().uniform(100, 255);
        int b = theRNG().uniform(100, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }

    // fill with color and display final result
    Mat dst = Mat::zeros(marker.size(), CV_8UC3);
    for (int row = 0; row < marker.rows; row++) {
        for (int col = 0; col < marker.cols; col++) {
            int index = marker.at<int>(row, col);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(row, col) = colors[index - 1];
            } else {
                dst.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
            }
        }
    }
    int dep = dst.depth();
    int chn = dst.channels();

    imshow("Final Result", dst);
//    std::vector<Mat> channels;
//    Mat imageBlueChannel;
//    split(dst, channels);//拆分
//    imageBlueChannel = channels.at(0);//蓝通道
//    imshow("imageBlueChannel Result", imageBlueChannel);
    Mat gray_out;
    cvtColor(dst, gray_out, COLOR_BGR2GRAY);
    imshow("gray_out Result", gray_out);
    Mat out_binary;
    threshold(gray_out, out_binary, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);
    imshow("out_binary Result", out_binary);
    Mat kernel = getStructuringElement(MORPH_RECT, Point(5, 5));
    morphologyEx(out_binary, out_binary, MORPH_CLOSE, kernel);
    imshow("out_binary xx Result", out_binary);
    vector<vector<Point>> contours_;
    vector<Vec4i> hireachy;
    int width = dst.cols;
    int height = dst.rows;
    cout << "width : " << width << "  height: " << height << endl;
    Mat drawImage = Mat::zeros(src.size(), CV_8UC3);
    findContours(out_binary, contours_, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    for (size_t t = 0; t < contours_.size(); t++) {
        Rect rect = boundingRect(contours_[t]);
//        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
        drawContours(drawImage, contours_, static_cast<int>(t), Scalar(0, 0, 255), 2, CV_AA, hireachy, 0, Point());
    }
    imshow("contours", drawImage);
    waitKey(0);
    return 0;
}

int main(int argc, char **argv) {
    Mat src_image;
//        src_image = imread("1.jpg");
//    src_image = imread("933.jpg");
//    src_image = imread("2.jpg");
//    src_image = imread("1127.jpg");
//    src_image = imread("1593422907134.jpg");
//    src_image = imread("1593423129890.jpg");
//    src_image = imread("1593421145441.jpg");
//    src_image = imread("IMG_20200628_173209.jpg");
//    src_image = imread("1593423448512.jpg");
//    src_image = imread("1593681794972.jpg");
    src_image = imread("1593423011560.jpg");
    imshow("原图", src_image);
    showImage(src_image);


    return 0;
}