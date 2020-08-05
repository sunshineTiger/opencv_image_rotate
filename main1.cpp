#include <iostream>
#include <opencv2/opencv.hpp>


#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp> //图像处理头文件

#include <cv.h>

#include <highgui.h>

#include <iostream>

using namespace cv;

using namespace std;

//根据三个点计算中间那个点的夹角，这里是用三角变换，感兴趣的可以求一下公式的推导 pt1 pt0 pt2
Point2f srcTri[4], dstTri[4];
int clickTimes = 0;  //在图像上单击次数
//主函数，小编这里反省一下，没有把一些函数封装起来，显得比较乱，小编建议大家养成把处理得函数都封装起来，主函数就不会像小编这样看上去很乱了。

double getAngle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {

    double dx1 = pt1.x - pt0.x;

    double dy1 = pt1.y - pt0.y;

    double dx2 = pt2.x - pt0.x;

    double dy2 = pt2.y - pt0.y;

    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);

}

//寻找最大边框

int findLargestSquare(const vector<vector<cv::Point> > &squares, vector<cv::Point> &biggest_square) {

    if (!squares.size()) return -1;

    int max_width = 0;

    int max_height = 0;

    int max_square_idx = 0;

    for (int i = 0; i < squares.size(); i++) {

        cv::Rect rectangle = boundingRect(Mat(squares[i]));

        if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {

            max_width = rectangle.width;

            max_height = rectangle.height;

            max_square_idx = i;

        }

    }

    biggest_square = squares[max_square_idx];

    return max_square_idx;

}

vector<vector<cv::Point> > contours, squares, hulls;
vector<cv::Point> hull, approx;
vector<cv::Point> largest_square;

void buildImage(Mat src) {

    Mat dst, gray, binarization, open, dilate_img, gauss_img, edges_img;
//    Mat src = imread("1.jpg");

//处理为灰度图

    cvtColor(src, gray, CV_BGR2GRAY);

//高斯滤波

//    GaussianBlur(gray, gauss_img, Size(3, 3), 2, 2);
medianBlur(gray,gauss_img,15);
    imshow("GaussianBlur", gauss_img);

//边缘检测

    Canny(gauss_img, edges_img, 100, 300, 3);
    imshow("xxxxxxx", edges_img);
//膨胀

    dilate(edges_img, dilate_img, Mat(), cv::Point(-1, -1), 2, 1, 1);
    imshow("dilate_img", dilate_img);
//定义容器类型得变量，不太明白得可以看看C++的容器这一部分知识


    contours.clear();
    squares.clear();
    hulls.clear();
//寻找出所有闭合的边框

    findContours(dilate_img, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);


    hull.clear();
    approx.clear();
//筛选边框

    for (int i = 0; i < contours.size(); i++) {

//获得轮廓的凸包

        convexHull(contours[i], hull);

//多边形拟合凸包边框(此时的拟合的精度较低)

//arcLength得到轮廓的长度

        approxPolyDP(Mat(hull), approx, arcLength(Mat(hull), true) * 0.02, true);

//筛选出四边形的各个角度都接近直角的凸四边形

        if (approx.size() == 4 && isContourConvex(Mat(approx))) {

//            double maxCosine = 0;

//            for (int j = 2; j < 5; j++) {
//
//                double cosine = fabs(getAngle(approx[j % 4], approx[j - 2], approx[j - 1]));
//
//                maxCosine = MAX(maxCosine, cosine);
//
//            }


//            if (maxCosine < 0.8) {

                squares.push_back(approx);

                hulls.push_back(hull);

//            }

        }

    }



//找出外接矩形最大的四边形
    largest_square.clear();
    std::cerr <<"squares大小:"<<squares.size()<< std::endl;
    int idex = findLargestSquare(squares, largest_square);
    if (idex <= 0 || squares[idex].empty()) {
        std::cerr << "picture is mark failed" << std::endl;
        return;
    }

    for (int i = 0; i < 4; i++) {
//        srcTri[i].x = squares[idex][i].x;
//        srcTri[i].y = squares[idex][i].y;
//        if (squares.size() > 0 && idex <= squares.size() - 1)
//            circle(src, squares[idex][i], 5, Scalar(0, 0, 255), 5);
//        printf("x:%d y:%d\n", squares[idex][i].x, squares[idex][i].y);
        if (i < 3) {

            line(src, squares[idex][i], squares[idex][i + 1], Scalar(0, 0, 255), 5);

        } else

            line(src, squares[idex][3], squares[idex][0], Scalar(0, 0, 255), 5);

    }

//显示出来

    namedWindow("draw", CV_WINDOW_NORMAL);

    imshow("draw", src);
//
//
//    int p1x = squares[idex][2].x;
//    int p1y = squares[idex][2].y;
//    int p2x = squares[idex][1].x;
//    int p2y = squares[idex][1].y;
//    int p3x = squares[idex][3].x;
//    int p3y = squares[idex][3].y;
//    double dis1 = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
//    double dis2 = sqrt((p1x - p3x) * (p1x - p3x) + (p1y - p3y) * (p1y - p3y));
//    int width = -1;
//    int height = -1;
//    if (dis1 > dis2) {
//        //ver竖图
//        double bili = dis1 / dis2;
//        if (bili <= 1.1) {
//            dstTri[0].x = 500;
//            dstTri[0].y = 500;
//            dstTri[1].x = 0;
//            dstTri[1].y = 500;
//            dstTri[2].x = 0;
//            dstTri[2].y = 0;
//            dstTri[3].x = 500;
//            dstTri[3].y = 0;
//            width = 400;
//            height = 400;
//        } else {
//            int my_width =(int) dis2;
//            int my_height = (int) (dis2 * 1.7777);
//            dstTri[0].x = my_width;
//            dstTri[0].y = my_height;
//            dstTri[1].x = 0;
//            dstTri[1].y = my_height;
//            dstTri[2].x = 0;
//            dstTri[2].y = 0;
//            dstTri[3].x = my_width;
//            dstTri[3].y = 0;
//            width = my_width;
//            height = my_height;
//        }
//    } else if (dis1 < dis2) {
//        //hor横图
//        double bili = dis2 / dis1;
//        if (bili <= 1.1) {
//            dstTri[0].x = 500;
//            dstTri[0].y = 500;
//            dstTri[1].x = 0;
//            dstTri[1].y = 500;
//            dstTri[2].x = 0;
//            dstTri[2].y = 0;
//            dstTri[3].x = 500;
//            dstTri[3].y = 0;
//            width = 500;
//            height = 500;
//
//        } else {
//            int my_width = (int) (dis1 * 1.7777);
//            int my_height = (int) dis1;
//            dstTri[0].x = my_width;
//            dstTri[0].y = my_height;
//            dstTri[1].x = 0;
//            dstTri[1].y = my_height;
//            dstTri[2].x = 0;
//            dstTri[2].y = 0;
//            dstTri[3].x = my_width;
//            dstTri[3].y = 0;
//            width = my_width;
//            height = my_height;
//        }
//    } else {
//        //equ
//        dstTri[0].x = 500;
//        dstTri[0].y = 500;
//        dstTri[1].x = 0;
//        dstTri[1].y = 500;
//        dstTri[2].x = 0;
//        dstTri[2].y = 0;
//        dstTri[3].x = 500;
//        dstTri[3].y = 0;
//        width = 500;
//        height = 500;
//    }
//    if (width < 0 || height < 0)
//        return;
//    printf("-----width:%d ------height:%d\n", width, height);
////    dstTri[0].x=500;
////    dstTri[0].y=500;
////    dstTri[1].x=0;
////    dstTri[1].y=500;
////    dstTri[2].x=0;
////    dstTri[2].y=0;
////    dstTri[3].x=500;
////    dstTri[3].y=0;
//    cv::Mat resultimg;
//    cv::Mat warpmatrix = cv::getPerspectiveTransform(srcTri, dstTri);
//    if (dis1 > dis2) {
//        cv::warpPerspective(src, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
//
//    } else {
//        //hor横图
//        cv::warpPerspective(src, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
//    }
//
//    imshow("warpPerspective", resultimg);

}

int main(int argc, char *argv[]) {
    VideoCapture cap; //定义视频对象
    cap.open(0);  //0为设备的ID号
    Mat frame;
    for (;;) {
        cap >> frame;
        if (!frame.empty()) {
            buildImage(frame);
        }
        waitKey(30); //0.03s获取一帧
    }


    return 0;

}