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


int main(){
    clock_t start, finish;
//    VideoCapture cap; //定义视频对象
//    cap.open(0);  //0为设备的ID号
    Mat frame;
    Mat out;
//    for(;;)
//    {
//        cap >> frame;
//        if(frame.data)
//        {
////            imshow("frame",frame);
////            buildImage(frame);
//            findSquares(frame,out);
//        }
//        waitKey(30); //0.03s获取一帧
//    }
    frame=imread("2.jpg");
    Point2f srcPoints[4], dstPoints[4];
    cv::Mat bw;
    cv::Mat binPic;
    cv::Mat cannyPic;
    cv::Mat outPic;
    cv::Mat linePic;
    start = clock();
    //gray image 转换成灰度图
    cv::cvtColor(frame, bw, CV_BGR2GRAY);
    medianBlur(bw, bw, 7); //中值滤波
    threshold(bw, binPic, 80, 255, THRESH_BINARY); //阈值化为二值图片


    double  cannyThr = 200, FACTOR = 2.5;
    Canny(binPic, cannyPic, cannyThr, cannyThr*FACTOR);
    imshow("cannyPic",cannyPic);
    vector<vector<Point>> contours;    //储存轮廓
    vector<Vec4i> hierarchy;

    findContours(cannyPic, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    linePic = Mat::zeros(cannyPic.rows, cannyPic.cols, CV_8UC3);
    for (int index = 0; index < contours.size(); index++){
        drawContours(linePic, contours, index, Scalar(rand() & 255, rand() & 255, rand() & 255), 1, 8/*, hierarchy*/);
    }
    vector<vector<Point>> polyContours(contours.size());
    int maxArea = 0;
    for (int index = 0; index < contours.size(); index++){
        if (contourArea(contours[index]) > contourArea(contours[maxArea]))
            maxArea = index;
        approxPolyDP(contours[index], polyContours[index], 10, true);
    }
    Mat polyPic = Mat::zeros(frame.size(), CV_8UC3);
    drawContours(polyPic, polyContours, maxArea, Scalar(0,0,255/*rand() & 255, rand() & 255, rand() & 255*/), 2);
    imshow("polyPic",polyPic);
    vector<int>  hull;
    convexHull(polyContours[maxArea], hull, false);
    int clickTimes=0;
    for (int i = 0; i < hull.size(); ++i){
        circle(polyPic, polyContours[maxArea][i], 10, Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
        srcPoints[clickTimes].x=polyContours[maxArea][i].x;
        srcPoints[clickTimes].y=polyContours[maxArea][i].y;
        clickTimes++;
        printf("x:%d y:%d\n",polyContours[maxArea][i].x,polyContours[maxArea][i].y);
    }
    addWeighted(polyPic, 0.5, frame, 0.5, 0, frame);
    imshow("frame",frame);
    finish = clock();
    cout << "渲染一次使用时间：" << double(finish - start) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
//    dstPoints[0] = Point2f(0, 0);
//    dstPoints[1] = Point2f(frame.cols, 0);
//    dstPoints[2] = Point2f(frame.cols, frame.rows);
//    dstPoints[3] = Point2f(0, frame.rows);
    dstPoints[0].x=0;
    dstPoints[0].y=0;
    dstPoints[3].x=500;
    dstPoints[3].y=0;
    dstPoints[1].x=0;
    dstPoints[1].y=500;
    dstPoints[2].x=500;
    dstPoints[2].y=500;
    for (int i = 0; i < 4; i++){
        polyContours[maxArea][i] = Point2f(polyContours[maxArea][i].x * 4, polyContours[maxArea][i].y * 4); //恢复坐标到原图
    }
    //对四个点进行排序 分出左上 右上 右下 左下
//    bool sorted = false;
//    int n = 4;
//    while (!sorted){
//        for (int i = 1; i < n; i++){
//            sorted = true;
//            if (polyContours[maxArea][i-1].x > polyContours[maxArea][i].x){
//                swap(polyContours[maxArea][i-1], polyContours[maxArea][i]);
//                sorted = false;
//            }
//        }
//        n--;
//    }
//    if (polyContours[maxArea][0].y < polyContours[maxArea][1].y){
//        srcPoints[0] = polyContours[maxArea][0];
//        srcPoints[3] = polyContours[maxArea][1];
//    }
//    else{
//        srcPoints[0] = polyContours[maxArea][1];
//        srcPoints[3] = polyContours[maxArea][0];
//    }
//
//    if (polyContours[maxArea][9].y < polyContours[maxArea][10].y){
//        srcPoints[1] = polyContours[maxArea][2];
//        srcPoints[2] = polyContours[maxArea][3];
//    }
//    else{
//        srcPoints[1] = polyContours[maxArea][3];
//        srcPoints[2] = polyContours[maxArea][2];
//    }
    Mat transMat = getPerspectiveTransform(srcPoints, dstPoints);
//    warpPerspective(frame, outPic, transMat, frame.size());
        cv::warpPerspective(frame,outPic,transMat,Size (500,500),INTER_LINEAR);
    imshow("outPic",outPic);





//    findSquares(frame,out);
    waitKey();
    return  0;
}

