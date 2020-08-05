#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include<vector>
#include<algorithm>

using namespace cv;
using namespace std;
Mat dst, gray_src, inv_dis, src_image;
char input_image[] = "input image";
char output_image[] = "output image";
int width;
int height;

bool checkRect(vector<Vec4i> lines);

void maoPaoSort(vector<Vec4i> array, int len);

bool cmpX(Vec4i a, Vec4i b) ///cmp函数传参的类型不是vector<int>型，是vector中元素类型,即int型
{
    if (a[0] > b[0]) {
        return true;
    } else {
        return false;
    }

}

bool cmpX1(Vec4i a, Vec4i b) ///cmp函数传参的类型不是vector<int>型，是vector中元素类型,即int型
{
    if (a[0] > b[0]) {
        return false;
    } else {
        return true;
    }

}

bool cmpY(Vec4i a, Vec4i b) ///cmp函数传参的类型不是vector<int>型，是vector中元素类型,即int型
{
    if (a[1] > b[1]) {
        return true;
    } else {
        return false;
    }

}

int main(int argc, char **argv) {

//    src_image = imread("1.jpg");
    src_image = imread("933.jpg");
//    src_image = imread("2.jpg");
//    src_image = imread("1127.jpg");
//    src_image = imread("1593422907134.jpg");
//    src_image = imread("1593423129890.jpg");
//    src_image = imread("1593421145441.jpg");
//    src_image = imread("IMG_20200628_173209.jpg");
    if (src_image.empty()) {
        printf("colud not load image ..\n");
        return -1;
    }
    medianBlur(src_image, src_image, 3);
    Mat imageEnhance;
//    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    filter2D(src_image, imageEnhance, CV_8UC3, kernel);
    imshow("filter2D", imageEnhance);
//    imshow("inpuxxxxxt_image", src_image);
    // 二值化处理
    Mat binary;
    cvtColor(imageEnhance, gray_src, COLOR_BGR2GRAY);
    threshold(gray_src, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
//    imshow("binary image", binary);
    bitwise_not(binary, binary, Mat());
    imshow("binary image", binary);
    Mat structureElement = getStructuringElement(MORPH_RECT, Size(4, 4), Point(-1, -1));
    dilate(binary, binary, structureElement);
    imshow("dilate", binary);


    vector<vector<Point>> contours;
    vector<Vec4i> hireachy;
    width = src_image.cols;
    height = src_image.rows;
    cout << "width : " << width << "  height: " << height << endl;
    Mat drawImage = Mat::zeros(src_image.size(), CV_8UC3);
    findContours(binary, contours, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    for (size_t t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
//        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
        if (rect.width > (src_image.cols / 2) && rect.width < (src_image.cols - 5)) {
            drawContours(drawImage, contours, static_cast<int>(t), Scalar(0, 0, 255), 2, CV_AA, hireachy, 0, Point());
        }
    }
    imshow("contours", drawImage);

    // 绘制直线
    vector<Vec4i> lines;
    Mat contoursImg;
    int accu = min(width * 0.5, height * 0.5);
    cvtColor(drawImage, contoursImg, COLOR_BGR2GRAY);
    Mat linesImage = Mat::zeros(src_image.size(), CV_8UC3);
    HoughLinesP(contoursImg, lines, 1, CV_PI / 180.0, 50, 50, 0);
//    for (size_t t = 0; t < lines.size(); t++) {
//        Vec4i ln = lines[t];
//        line(linesImage, Point(ln[0], ln[1]), Point(ln[2], ln[3]), Scalar(0, 0, 255), 2, LINE_AA, 0);
//    }
//    imshow("linesImage", linesImage);
    Vec4i topLine, bottomLine, leftLine, rightLine;
//    Vec4i line1, line2, line3, line4;
    Vec2i minXPoint, maxXPoint2, minPoint1, maxPoint21;
    minXPoint[0] = width;
    minXPoint[1] = height;
    maxXPoint2[0] = 0;
    maxXPoint2[1] = 0;
    minPoint1[0] = width;
    minPoint1[1] = height;
    maxPoint21[0] = 0;
    maxPoint21[1] = 0;

    if (checkRect(lines)) {
        //zuobianyigedian   lingxing
        sort(lines.begin(), lines.end(), cmpX1);
        for (int i = 0; i < lines.size(); i++) {
            Vec4i ln = lines[i];
//            cout << "ln  : " << ln << endl;
            int startX = ln[0];
            int startY = ln[1];
            int endX = ln[2];
            int endY = ln[3];
            if (endX > maxXPoint2[0] && endY >= startY) {
                Vec2i tempPoint(startX, startY);
                maxXPoint2 = tempPoint;
                if ((endY - startY) >= 0) {
                    //菱形右上边线
                    rightLine = ln;
                }

            }
        }
        for (int i = 0; i < lines.size(); i++) {
            Vec4i ln = lines[i];
//            cout << "ln  : " << ln << endl;
            int startX = ln[0];
            int startY = ln[1];
            int endX = ln[2];
            int endY = ln[3];
            if (endX > maxPoint21[0] && endY <= startY) {
                Vec2i tempPoint(startX, startY);
                maxPoint21 = tempPoint;
                if ((startY - endY) >= 0) {
                    //
                    bottomLine = ln;
                }

            }
        }

        sort(lines.begin(), lines.end(), cmpX);
        for (int i = 0; i < lines.size(); i++) {
            Vec4i ln = lines[i];
//            cout << "ln  : " << ln << endl;
            int startX = ln[0];
            int startY = ln[1];
            int endX = ln[2];
            int endY = ln[3];
            if (startX < minXPoint[0] && endY <= startY) {
                Vec2i tempPoint(startX, startY);
                minXPoint = tempPoint;
//                cout << "startY - endY  : " << startY - endY << endl;
                if ((startY - endY) >= 0) {

                    topLine = ln;
                }
            }
        }
        for (int i = 0; i < lines.size(); i++) {
            Vec4i ln = lines[i];
//            cout << "ln  : " << ln << endl;
            int startX = ln[0];
            int startY = ln[1];
            int endX = ln[2];
            int endY = ln[3];
            if (startX < minPoint1[0] && endY >= startY) {
                Vec2i tempPoint(startX, startY);
                minPoint1 = tempPoint;
//                cout << "startY - endY  : " << startY - endY << endl;
                if ((endY - startY) >= 0) {

                    leftLine = ln;
                }
            }
        }
    } else {
        //zuobianlianggedian   changfangxing
        topLine = Vec4i(width / 2, height / 2, width / 2, height / 2);
        bottomLine = Vec4i(width / 2, height / 2, width / 2, height / 2);
        leftLine = Vec4i(width / 2, height / 2, width / 2, height / 2);
        rightLine = Vec4i(width / 2, height / 2, width / 2, height / 2);
        for (int i = 0; i < lines.size(); i++) {
            Vec4i ln = lines[i];
//            cout << "ln  : " << ln << endl;
            int startX = ln[0];
            int startY = ln[1];
            int endX = ln[2];
            int endY = ln[3];
            if (startY <= height / 2.0 && endY <= height / 2.0 && abs(startY - endY) < 50) {
                if (topLine[1] > startY && topLine[3] > endY)
                    topLine = lines[i];
            }
            if (startY > height / 2.0 && endY > height / 2.0 && abs(startY - endY) < 50) {
                if (bottomLine[1] < startY && bottomLine[3] < endY)
                    bottomLine = lines[i];
            }
            if (startX <= width / 2.0 && endX <= width / 2.0 && abs(startX - endX) < 50) {
                if (leftLine[0] > startX && leftLine[2] > endX)
                    leftLine = lines[i];
            }
            if (startX > width / 2.0 && endX > width / 2.0 && abs(startX - endX) < 50) {
                if (rightLine[0] < startX && rightLine[2] < endX)
                    rightLine = lines[i];
            }
        }

    }
    cout << "--->>>>  : " << maxXPoint2 << endl;

    cout << "topLine : " << topLine << endl;
    cout << "bottomLine : " << bottomLine << endl;
    cout << "leftLine : " << leftLine << endl;
    cout << "rightLine : " << rightLine << endl;


    // 拟合四条直线方程
    float k1, c1;
    k1 = float(topLine[3] - topLine[1]) / float(topLine[2] - topLine[0]);
    c1 = topLine[1] - k1 * topLine[0];
    float k2, c2;
    k2 = float(bottomLine[3] - bottomLine[1]) / float(bottomLine[2] - bottomLine[0]);
    c2 = bottomLine[1] - k2 * bottomLine[0];
    float k3, c3;
    if (leftLine[2] - leftLine[0] == 0)
        k3 = 255;
    else
        k3 = float(leftLine[3] - leftLine[1]) / float(leftLine[2] - leftLine[0]);
    c3 = leftLine[1] - k3 * leftLine[0];
    float k4, c4;
    if (rightLine[2] - rightLine[0] == 0)
        k4 = 255;
    else
        k4 = float(rightLine[3] - rightLine[1]) / float(rightLine[2] - rightLine[0]);
    c4 = rightLine[1] - k4 * rightLine[0];

    // 四条直线交点
    Point p1; // 左上角
    p1.x = static_cast<int>((c1 - c3) / (k3 - k1));
    p1.y = static_cast<int>(k1 * p1.x + c1);
    Point p2; // 右上角
    p2.x = static_cast<int>((c1 - c4) / (k4 - k1));
    p2.y = static_cast<int>(k1 * p2.x + c1);
    Point p3; // 左下角
    p3.x = static_cast<int>((c2 - c3) / (k3 - k2));
    p3.y = static_cast<int>(k2 * p3.x + c2);
    Point p4; // 右下角
    p4.x = static_cast<int>((c2 - c4) / (k4 - k2));
    p4.y = static_cast<int>(k2 * p4.x + c2);
    cout << "p1(x, y)=" << p1.x << "," << p1.y << endl;
    cout << "p2(x, y)=" << p2.x << "," << p2.y << endl;
    cout << "p3(x, y)=" << p3.x << "," << p3.y << endl;
    cout << "p4(x, y)=" << p4.x << "," << p4.y << endl;

    // 显示四个点坐标
    circle(linesImage, p1, 2, Scalar(255, 0, 0), 2, 8, 0);
    circle(linesImage, p2, 2, Scalar(255, 0, 0), 2, 8, 0);
    circle(linesImage, p3, 2, Scalar(255, 0, 0), 2, 8, 0);
    circle(linesImage, p4, 2, Scalar(255, 0, 0), 2, 8, 0);
//    line(linesImage, Point(topLine[0], topLine[1]), Point(topLine[2], topLine[3]), Scalar(0, 255, 0), 2, 8, 0);
    imshow("four corners", linesImage);

    // 透视变换
    vector<Point2f> src_corners(4); // 原来的点
    src_corners[0] = p1;
    src_corners[1] = p2;
    src_corners[2] = p3;
    src_corners[3] = p4;

    vector<Point2f> dst_corners(4); // 目标点位
    dst_corners[0] = Point(0, 0);
    dst_corners[1] = Point(width, 0);
    dst_corners[2] = Point(0, height);
    dst_corners[3] = Point(width, height);

    // 获取变换矩阵
    Mat reslutImg;
    Mat warpmatrix = getPerspectiveTransform(src_corners, dst_corners);
    warpPerspective(src_image, reslutImg, warpmatrix, reslutImg.size(), INTER_LINEAR);
    imshow(output_image, reslutImg);
    waitKey(0);
    return 0;
}


bool checkRect(vector<Vec4i> templines) {
    bool rect = false;
    vector<Vec4i> lines = templines;
    Vec4i line1, line2, line3, line4;
    Vec2i minXPoint, maxXPoint2, minPoint1, maxPoint21;
    minXPoint[0] = width;
    minXPoint[1] = height;
    maxXPoint2[0] = 0;
    maxXPoint2[1] = 0;
    minPoint1[0] = width;
    minPoint1[1] = height;
    maxPoint21[0] = 0;
    maxPoint21[1] = 0;
    float tempK;


    sort(lines.begin(), lines.end(), cmpX1);

    for (int i = 0; i < lines.size(); i++) {
        Vec4i ln = lines[i];
        int startX = ln[0];
        int startY = ln[1];
        int endX = ln[2];
        int endY = ln[3];
        if (endX > maxPoint21[0]) {
            Vec2i tempPoint(startX, startY);
            maxPoint21 = tempPoint;

            tempK = (float(endY) - float(startY)) / (float(endX) - float(startX));

            if (tempK > 0) {
                if (tempK > 1.5) {
                    //rect
                    rect = false;
                } else {
                    //lingxing
                    rect = true;
                }
            } else {
                if (tempK < -1.5) {
                    //rect
                    rect = false;
                } else {
                    //lingxing
                    rect = true;
                }

            }
            line3 = ln;
        }
    }
    cout << "tempK--------------   :  " << tempK << endl;
    cout << "xian-----------------------" << line1 << endl;
    cout << "xian -----------------------" << line2 << endl;
    cout << "xian -----------------------" << line3 << endl;
    line(src_image, Point(line1[0], line1[1]), Point(line1[2], line1[3]), Scalar(0, 0, 255),
         2,
         LINE_AA,
         0);//white
    line(src_image, Point(line2[0], line2[1]), Point(line2[2], line2[3]), Scalar(0, 255, 0),
         2,
         LINE_AA, 0);//white
    line(src_image, Point(line3[0], line3[1]), Point(line3[2], line3[3]), Scalar(255, 0, 0),
         2,
         LINE_AA, 0);//white
    circle(src_image, Point(line1[0], line1[1]),
           5, Scalar(106, 217, 255), 16);
    circle(src_image, Point(line1[2], line1[3]),
           5, Scalar(140, 49, 255), 16);
    circle(src_image, Point(line2[0], line2[1]),
           5, Scalar(106, 217, 255), 16);
    circle(src_image, Point(line2[2], line2[3]),
           5, Scalar(140, 49, 255), 16);
    circle(src_image, Point(line3[0], line3[1]),
           5, Scalar(106, 217, 255), 16);
    circle(src_image, Point(line3[2], line3[3]),
           5, Scalar(140, 49, 255), 16);
    imshow("xxxxx", src_image);
    if (rect) {
        cout << "ling xing -----------------------" << endl;
        return true;

    } else {
        cout << "shui ping -----------------------" << endl;
        return false;
    }
}
