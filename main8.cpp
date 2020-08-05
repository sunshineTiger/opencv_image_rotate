#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
vector<cv::Point> approx;
vector<vector<cv::Point> > contours;
vector<Vec4i> hierarchy;
vector<cv::Point> largest_square;
Point2f srcTri[4], dstTri[4];
vector<vector<Point> > squares;

void buildImage(Mat &frame) {
      int max_width = 0;
       int max_height = 0;
       largest_square.clear();
       Mat gray, binary,canny,Blur,otsu,gray_one;
       gray_one = Mat(frame.size(), CV_8U);
   //    cvtColor(frame, gray, COLOR_BGRA2GRAY);//灰度
   //    imshow("gray", gray);

   //    GaussianBlur(gray, Blur, Size(3, 3), 5, 5);//滤波
       medianBlur(frame, Blur, 9);
       imshow("GaussianBlur", Blur);

       int ch[] = {1, 0};
       mixChannels(&Blur, 1, &gray_one, 1, ch, 1);
       imshow("mixChannels", gray_one);
   //    threshold(gray_one, binary, 127, 255, THRESH_BINARY);//二值
   //    imshow("binary", binary);

       Canny(gray_one, canny, 150, 450, 3);//canny边缘检测
       imshow("canny", canny);

       //膨脹
       dilate(canny, canny, Mat(), Point(-1, -1));
       imshow("dilate", canny);
   //    threshold(binary,otsu,0,255,THRESH_OTSU);
   //    imshow("otsu", otsu);

       findContours(canny, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);
       // 检测所找到的轮廓
       for (size_t i = 0; i < contours.size(); i++) {
           //使用图像轮廓点进行多边形拟合
   //        if (arcLength(Mat(contours[i]), true) < 1000)
   //            continue;
           approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
           std::cout << "picture is mark " <<approx.size() << std::endl;
           //计算轮廓面积后，得到矩形4个顶点
           if (approx.size() == 4 && fabs(contourArea(Mat(approx))) >= 500 && isContourConvex(Mat(approx))) {

               if (approx.empty())
                   continue;
               const Point *p = &approx[0];
               if (p->x > 3 && p->y > 3) {
                   cv::Rect rectangle = boundingRect(Mat(approx));

                   if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {

                       max_width = rectangle.width;

                       max_height = rectangle.height;

                   }
               }
               largest_square = approx;
           }
       }
       //    int idex = findLargestSquare(squares, largest_square);
       if (largest_square.empty()) {
           std::cerr << "picture is mark failed" << std::endl;
           return;
       }
       const Point *p = &largest_square[0];
       int n = (int) largest_square.size();
       polylines(frame, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);

       for (int i = 0; i < 4; i++) {
           srcTri[i].x = largest_square[3 - i].x;
           srcTri[i].y = largest_square[3 - i].y;
           if (!largest_square.empty()) {
               circle(frame, largest_square[i], 5, Scalar(0, 0, 255), 5);
   //            printf("x:%d y:%d\n", squares[idex][i].x, squares[idex][i].y);
           }
       }
       imshow("dst", frame);
       int p1x = largest_square[2].x;
       int p1y = largest_square[2].y;
       int p2x = largest_square[1].x;
       int p2y = largest_square[1].y;
       int p3x = largest_square[3].x;
       int p3y = largest_square[3].y;
       double dis1 = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
       double dis2 = sqrt((p1x - p3x) * (p1x - p3x) + (p1y - p3y) * (p1y - p3y));
       int width = -1;
       int height = -1;
       if (dis1 > dis2) {
           //ver竖图
   //        printf("ver竖图\n");
           double bili = dis1 / dis2;
           if (bili <= 1.1) {
               dstTri[0].x = 500;
               dstTri[0].y = 500;
               dstTri[1].x = 0;
               dstTri[1].y = 500;
               dstTri[2].x = 0;
               dstTri[2].y = 0;
               dstTri[3].x = 500;
               dstTri[3].y = 0;
               width = 400;
               height = 400;
           } else {
               int my_width = 480;
               int my_height = 640;
               dstTri[0].x = my_width;
               dstTri[0].y = my_height;
               dstTri[1].x = 0;
               dstTri[1].y = my_height;
               dstTri[2].x = 0;
               dstTri[2].y = 0;
               dstTri[3].x = my_width;
               dstTri[3].y = 0;
               width = my_width;
               height = my_height;
           }
       } else if (dis1 < dis2) {
           //hor横图
   //        printf("hor横图\n");
           double bili = dis2 / dis1;
           if (bili <= 1.1) {
               dstTri[0].x = 500;
               dstTri[0].y = 500;
               dstTri[1].x = 0;
               dstTri[1].y = 500;
               dstTri[2].x = 0;
               dstTri[2].y = 0;
               dstTri[3].x = 500;
               dstTri[3].y = 0;
               width = 500;
               height = 500;

           } else {
               int my_width = 640;
               int my_height = 480;
               dstTri[0].x = my_width;
               dstTri[0].y = my_height;
               dstTri[1].x = 0;
               dstTri[1].y = my_height;
               dstTri[2].x = 0;
               dstTri[2].y = 0;
               dstTri[3].x = my_width;
               dstTri[3].y = 0;
               width = my_width;
               height = my_height;
           }
       } else {
           //equ
           dstTri[0].x = 500;
           dstTri[0].y = 500;
           dstTri[1].x = 0;
           dstTri[1].y = 500;
           dstTri[2].x = 0;
           dstTri[2].y = 0;
           dstTri[3].x = 500;
           dstTri[3].y = 0;
           width = 500;
           height = 500;
       }
       if (width < 0 || height < 0)
           return;
   //    sortdisPorint(dstTri);
   //    sortdisPorint(srcTri);

       cv::Mat resultimg;
       cv::Mat warpmatrix = cv::getPerspectiveTransform(srcTri, dstTri);
       if (dis1 > dis2) {
           cv::warpPerspective(frame, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
       } else {
           //hor横图
           cv::warpPerspective(frame, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
       }
       imshow("warpPerspective", resultimg);
}

static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

void buildImage1(Mat &frame) {
    int max_width = 0;
    int max_height = 0;
    largest_square.clear();
    Mat gray, binary, canny, Blur, otsu, gray_one;
    gray_one = Mat(frame.size(), CV_8U);
    cvtColor(frame, gray, COLOR_BGRA2GRAY);//灰度
    imshow("gray", gray);

    //    GaussianBlur(gray, Blur, Size(3, 3), 5, 5);//滤波
    medianBlur(gray, Blur, 9);
//    pyrMeanShiftFiltering(gray, Blur, 10, 100);
    imshow("GaussianBlur", Blur);
    adaptiveThreshold(Blur, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 31, 6);//二值
//    adaptiveThreshold(binary, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 10);//二值
//    int ch[] = {2, 0};
//    mixChannels(&Blur, 1, &gray_one, 1, ch, 1);
//    imshow("mixChannels", gray_one);
//        threshold(Blur, binary, 127, 255, THRESH_BINARY);//二值
    imshow("binary", binary);

    Canny(binary, canny, 100, 300, 5);//canny边缘检测
    imshow("canny", canny);
//
//    //膨脹
//    dilate(canny, canny, Mat(), Point(-1, -1));
//    imshow("dilate", canny);
    //    threshold(binary,otsu,0,255,THRESH_OTSU);
    //    imshow("otsu", otsu);

//    findContours(canny, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    findContours(canny, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        //使用图像轮廓点进行多边形拟合
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

        //计算轮廓面积后，得到矩形4个顶点
        if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx))) {
//            double maxCosine = 0;

//            for (int j = 2; j < 5; j++) {
//                // 求轮廓边缘之间角度的最大余弦
//                double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
//                maxCosine = MAX(maxCosine, cosine);
//            }

//            if (maxCosine < 0.3) {
                squares.push_back(approx);
//            }
        }
    }
    for (size_t i = 0; i < squares.size(); i++) {
        const Point *p = &squares[i][0];

        int n = (int) squares[i].size();
        if (p->x > 3 && p->y > 3) {
            polylines(frame, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
        }
    }
    imshow("dst", frame);
}

void show1() {
//    Mat frame = imread("8.png");
    Mat frame = imread("933.jpg");
    imshow("frame", frame);
    buildImage1(frame);
    waitKey();
}

void show2() {
    VideoCapture cap; //定义视频对象
    cap.open(0); //0为设备的ID号
    Mat frame;
    for (;;) {
        cap >> frame;
        if (!frame.empty()) {
            imshow("frame", frame);
            buildImage1(frame);
        }
        waitKey(30); //0.03s获取一帧
    }
}

int main(int argc, char *argv[]) {
    show1();
//    show2();
    return 0;

}