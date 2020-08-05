#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Mat image;
int imageHeight;
bool photo = true;
int imageWidth;
vector<cv::Point> polylines_v;
cv::Point2f center(0, 0);

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b) {
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    float denom;

    if (float d = ((float) (x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))) {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    } else
        return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f> &corners,
                 cv::Point2f center) {
    std::vector<cv::Point2f> top, bot;

    for (int i = 0; i < corners.size(); i++) {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    corners.clear();

    if (top.size() == 2 && bot.size() == 2) {
        cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
        cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
        cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
        cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];


        corners.push_back(tl);
        corners.push_back(tr);
        corners.push_back(br);
        corners.push_back(bl);
    }
}


int getMat(int argc, char *argv[]) {
    //read image 载入原始图片
//    cv::Mat src = cv::imread("8.png");
    cv::Mat src = cv::imread("933.jpg");
    Mat cdst, dst;
    if (src.empty()) {
        std::cout << "src image is null!" << std::endl;
        return -1;
    }
    cv::Mat bw;
    //gray image 转换成灰度图
    cv::cvtColor(src, bw, CV_BGR2GRAY);

//    cv::imshow("gray", bw);
    //blur image  图像模糊降噪
    cv::blur(bw, bw, cv::Size(3, 3));
//    cv::imshow("blur", bw);
    //canny 函数边缘检测
//    cv::Canny(bw, bw, 100, 100, 3);
    cv::Canny(bw, bw, 100, 300, 3);
//    cv::Canny(src, dst,  200, 200, 3);
//    cv::cvtColor(dst, cdst, CV_GRAY2BGR);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(bw, lines, 1, CV_PI / 180, 70, 30, 10);
//    cv::HoughLinesP(bw, lines, 1, CV_PI/180, 50, 50, 10);

    // Expand the lines
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i v = lines[i];
        line(src, Point(v[0], v[1]), Point(v[0], v[1]), Scalar(0, 0, 255), 3, CV_AA);
//        lines[i][0] = 0;
//        lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * -v[0] + v[1];s
//        lines[i][2] = src.cols;
//        lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * (src.cols - v[2]) + v[3];
    }
    imshow("source", src);
    imshow("detected lines", bw);
    waitKey();
    return 0;
    /*
    std::vector<cv::Point2f> corners;
    for (int i = 0; i < lines.size(); i++)
    {
        for (int j = i+1; j < lines.size(); j++)
        {
            cv::Point2f pt = computeIntersect(lines[i], lines[j]);
            if (pt.x >= 0 && pt.y >= 0)
                corners.push_back(pt);
        }
    }

    std::vector<cv::Point2f> approx;
    cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true) * 0.02, true);
    cv::imshow("Canny", bw);
    if (approx.size() != 4)
    {
        std::cout << "The object is not quadrilateral!" << std::endl;    cv::waitKey();
        return -1;
    }

    // Get mass center
    for (int i = 0; i < corners.size(); i++)
        center += corners[i];
    center *= (1. / corners.size());

    sortCorners(corners, center);
    if (corners.size() == 0){
        std::cout << "The corners were not sorted correctly!" << std::endl;    cv::waitKey();
        return -1;
    }
    cv::Mat dst = src.clone();

    // Draw lines
    for (int i = 0; i < lines.size(); i++)
    {
        cv::Vec4i v = lines[i];
        cv::line(dst, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0,255,0));
    }

    // Draw corner points
    cv::circle(dst, corners[0], 3, CV_RGB(255,0,0), 2);
    cv::circle(dst, corners[1], 3, CV_RGB(0,255,0), 2);
    cv::circle(dst, corners[2], 3, CV_RGB(0,0,255), 2);
    cv::circle(dst, corners[3], 3, CV_RGB(255,255,255), 2);

    // Draw mass center
    cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);

    cv::Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);

    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));

    cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
    cv::warpPerspective(src, quad, transmtx, quad.size());

    cv::imshow("image", dst);
    cv::imshow("quadrilateral", quad);
    cv::waitKey();
    return 0;*/
}

void onMouse(int event, int x, int y, int flags, void *utsc);

Point2f srcTri[4], dstTri[4];
int clickTimes = 0;  //在图像上单击次数
Mat imageWarp;
bool over = false;

void onMouse(int event, int x, int y, int flags, void *utsc) {
    if (event == CV_EVENT_LBUTTONUP)   //响应鼠标左键抬起事件
    {
        photo = false;
        circle(image, Point(x, y), 2.5, Scalar(0, 0, 255), 2.5);  //标记选中点
        imshow("Source Image", image);
        srcTri[clickTimes].x = x;
        srcTri[clickTimes].y = y;
        clickTimes++;
        printf("x:%d,y:%d\n", x, y);
    }
    if (clickTimes == 4 && !over) {
        over = true;
        printf("click finish\n");
        cv::Mat resultimg;
        //左上右上左下右下
        int width = 480;
        int height = 640;
        dstTri[0].x = 0;
        dstTri[0].y = 0;
        dstTri[1].x = width;
        dstTri[1].y = 0;
        dstTri[2].x = 0;
        dstTri[2].y = height;
        dstTri[3].x = width;
        dstTri[3].y = height;
        Mat Blur_image;
        medianBlur(image, Blur_image, 3);
        Mat result, bg, fg;
     int   line_width=50;
        Rect rect(line_width, line_width, imageWidth - line_width*2, imageHeight - line_width*2);//左上坐标（X,Y）和长宽
        grabCut(Blur_image, result, rect, bg, fg, 1, GC_INIT_WITH_RECT);
//        imshow("grab", result);
        /*threshold(result, result, 2, 255, CV_THRESH_BINARY);
        imshow("threshold", result);*/

        compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255
        imshow("result", result);
        Mat foreground(image.size(), CV_8UC3, Scalar(0, 0, 0));
        image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
        imshow("foreground", foreground);

        vector<vector<Point>> contours_;
        vector<Vec4i> hierarcy;
        Mat binary;
        cvtColor(foreground, foreground, CV_BGR2GRAY);
        threshold(foreground, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
//        bitwise_not(binary, binary, Mat());
//        imshow("bitwise_notimage", binary);
        findContours(binary, contours_, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        Mat drawImage = Mat::zeros(image.size(), CV_8UC3);
//        imshow("binary image", binary);
        findContours(binary, contours_, hierarcy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
        for (size_t t = 0; t < contours_.size(); t++) {
            Rect rect = boundingRect(contours_[t]);
//        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
            if (rect.width > (image.cols / 2) && rect.width < (image.cols - 5)) {
                drawContours(drawImage, contours_, static_cast<int>(t), Scalar(0, 0, 255), 2, CV_AA, hierarcy, 0,
                             Point());
            }
        }
        imshow("contours", drawImage);


        vector<Rect> boundRect(contours_.size());  //定义外接矩形集合
        vector<RotatedRect> box(contours_.size()); //定义最小外接矩形集合
        Point2f rect1[4];
        Mat dstImg = image.clone();
        for (int i = 0; i < contours_.size(); i++) {
            box[i] = minAreaRect(Mat(contours_[i]));  //计算每个轮廓最小外接矩形
            boundRect[i] = boundingRect(Mat(contours_[i]));
            circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
            box[i].points(rect1);  //把最小外接矩形四个端点复制给rect数组
            rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y),
                      Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height),
                      Scalar(0, 255, 0), 2, 8);
            for (int j = 0; j < 4; j++) {
                line(dstImg, rect1[j], rect1[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
            }
        }
        imshow("dst", dstImg);
////        Mat imageEnhance;
////        Mat src_gray;
////        cvtColor(foreground, src_gray, CV_BGR2GRAY);
////        Mat x_grad, y_grad;
////        Mat out_image;
////        Sobel(src_gray, x_grad, CV_16S, 1, 0, 3);
////        Sobel(src_gray, y_grad, CV_16S, 0, 1, 3);
////        convertScaleAbs(x_grad, x_grad);
////        convertScaleAbs(y_grad, y_grad);
////        imshow("x_grad", x_grad);
////        imshow("y_grad", y_grad);
////      addWeighted(x_grad,0.5,y_grad,0.5,0,out_image);
//////Laplacian(src_gray,out_image,-1);
//////        convertScaleAbs(out_image, out_image);
//////        imshow("addWeighted", out_image);
//////        Mat gray_src;
//////        cvtColor(imageEnhance, gray_src, COLOR_BGR2GRAY);
////        vector<vector<Point>> contours;
////        vector<Vec4i> hireachy;
////        width = image.cols;
////        height = image.rows;
////        Mat binary;
////    threshold(out_image, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
////        cout << "width : " << width << "  height: " << height << endl;
////        Mat drawImage = Mat::zeros(image.size(), CV_8UC3);
////        imshow("binary image", binary);
////        findContours(binary, contours, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
////        for (size_t t = 0; t < contours.size(); t++) {
////            Rect rect = boundingRect(contours[t]);
//////        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
////            if (rect.width > (image.cols / 2) && rect.width < (image.cols - 5)) {
////                drawContours(drawImage, contours, static_cast<int>(t), Scalar(0, 0, 255), 2, CV_AA, hireachy, 0,
////                             Point());
////            }
////        }
////        imshow("contours", drawImage);
////        Mat result, bg, fg;
////        Rect rect(srcTri[0].x,srcTri[0].y,srcTri[1].x-srcTri[0].x,srcTri[2].y-srcTri[0].y);//左上坐标（X,Y）和长宽
////        grabCut(image, result, rect, bg, fg, 1, GC_INIT_WITH_RECT);
////        imshow("grab", result);
////        /*threshold(result, result, 2, 255, CV_THRESH_BINARY);
////        imshow("threshold", result);*/
////
////        compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255
////        imshow("result",result);
////        Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
////        image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
////        imshow("foreground", foreground);
////        cv::Mat warpmatrix=cv::getPerspectiveTransform(srcTri,dstTri);
////        cv::warpPerspective(image,resultimg,warpmatrix,Size (width,height),INTER_LINEAR);
//////        cv::warpPerspective(image,resultimg,warpmatrix,resultimg.size(),INTER_LINEAR);
////        cv::imshow("resultimg",resultimg);
////        cv::imwrite("hello.jpg",resultimg);
////        Mat transform=Mat::zeros(3,3,CV_32FC1); //透视变换矩阵
////        transform=getPerspectiveTransform(srcTri,dstTri);  //获取透视变换矩阵
////        warpPerspective(image,imageWarp,transform,Size(image.rows,image.cols));  //透视变换
////        imshow("After WarpPerspecttive",imageWarp);
    }
}
//void onMouse(int event, int x, int y, int flags, void *utsc) {
//    if (event == CV_EVENT_LBUTTONUP)   //响应鼠标左键抬起事件
//    {
//        photo = false;
//        circle(image, Point(x, y), 2.5, Scalar(0, 0, 255), 2.5);  //标记选中点
//        imshow("Source Image", image);
//        srcTri[clickTimes].x = x;
//        srcTri[clickTimes].y = y;
//        clickTimes++;
//        printf("x:%d,y:%d\n", x, y);
//    }
//    if (clickTimes == 4 && !over) {
//        over = true;
//        printf("click finish\n");
//        cv::Mat resultimg;
//        //左上右上左下右下
////        int width = 480;
////        int height = 640;
////        dstTri[0].x = 0;
////        dstTri[0].y = 0;
////        dstTri[1].x = width;
////        dstTri[1].y = 0;
////        dstTri[2].x = 0;
////        dstTri[2].y = height;
////        dstTri[3].x = width;
////        dstTri[3].y = height;
//        Mat caijian_image;
////        Mat Blur_image;
////        medianBlur(image, Blur_image, 3);
//        Mat result, bg, fg;
//        int line_width = 100;
//        caijian_image=Mat(image,Rect(line_width, line_width,imageWidth - line_width * 2, imageHeight - line_width * 2));
////
//        imshow("caijian_image",caijian_image);
//
////        Rect rect(line_width, line_width,imageWidth - line_width * 2, imageHeight - line_width * 2);//左上坐标（X,Y）和长宽
//        Rect rect(5, 5, caijian_image.cols-5*2, caijian_image.rows-5*2);//左上坐标（X,Y）和长宽
//        grabCut(caijian_image, result, rect, bg, fg, 1, GC_INIT_WITH_RECT);
////        imshow("grab", result);
//        /*threshold(result, result, 2, 255, CV_THRESH_BINARY);
//        imshow("threshold", result);*/
//
//        compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255
//        imshow("result", result);
//        Mat foreground(caijian_image.size(), CV_8UC3, Scalar(0, 0, 0));
//        caijian_image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
//        imshow("foreground", foreground);
//        Mat src_gray;
//        Mat binary;
//        cvtColor(foreground, src_gray, CV_BGR2GRAY);
//        Mat x_grad, y_grad;
//        Mat out_image;
//        Mat Blur_image;
//        medianBlur(src_gray, Blur_image, 3);
//        Sobel(Blur_image, x_grad, CV_16S, 1, 0, 3);
//        Sobel(Blur_image, y_grad, CV_16S, 0, 1, 3);
//        convertScaleAbs(x_grad, x_grad);
//        convertScaleAbs(y_grad, y_grad);
//        imshow("x_grad", x_grad);
//        imshow("y_grad", y_grad);
//        addWeighted(x_grad, 0.5, y_grad, 0.5, 0, out_image);
//        imshow("addWeightedxx", out_image);
//
//        threshold(out_image, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
//        vector<vector<Point>> contours_;
//        vector<Vec4i> hierarcy;
//
//        bitwise_not(binary, binary, Mat());
//        imshow("bitwise_notimage", binary);
//        findContours(binary, contours_, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        vector<Rect> boundRect(contours_.size());  //定义外接矩形集合
//        vector<RotatedRect> box(contours_.size()); //定义最小外接矩形集合
//        Point2f rect1[4];
//        Mat dstImg = caijian_image.clone();
//        for (int i = 0; i < contours_.size(); i++) {
//            box[i] = minAreaRect(Mat(contours_[i]));  //计算每个轮廓最小外接矩形
//            boundRect[i] = boundingRect(Mat(contours_[i]));
//            circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
//            box[i].points(rect1);  //把最小外接矩形四个端点复制给rect数组
//            rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y),
//                      Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height),
//                      Scalar(0, 255, 0), 2, 8);
//            for (int j = 0; j < 4; j++) {
//                line(dstImg, rect1[j], rect1[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
//            }
//        }
//        imshow("dst", dstImg);
////        Mat Blur_image;
////        medianBlur(foreground, Blur_image, 9);
////        imshow("Blur_imageyxxx", Blur_image);
////        Mat imageEnhance;
////        Mat src_gray;
////        cvtColor(Blur_image, src_gray, CV_BGR2GRAY);
////        imshow("src_grayxxx", src_gray);
////        Mat x_grad, y_grad;
////        Mat out_image;
////
////        Sobel(src_gray, x_grad, CV_16S, 1, 0, 3);
////        Sobel(src_gray, y_grad, CV_16S, 0, 1, 3);
////        convertScaleAbs(x_grad, x_grad);
////        convertScaleAbs(y_grad, y_grad);
////        imshow("x_grad", x_grad);
////        imshow("y_grad", y_grad);
////        addWeighted(x_grad, 0.5, y_grad, 0.5, 0, out_image);
////        Mat canny_image;
////        Canny(out_image,canny_image,200,400,3,false);
////////Laplacian(src_gray,out_image,-1);
//////        convertScaleAbs(out_image, out_image);
////        imshow("addWeighted", out_image);
////        imshow("canny_image", canny_image);
////        Mat gray_src;
////        cvtColor(imageEnhance, gray_src, COLOR_BGR2GRAY);
//        vector<vector<Point>> contours;
//        vector<Vec4i> hireachy;
////        width = image.cols;
////        height = image.rows;
////        Mat binary;
////        threshold(src_gray, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
////        cout << "width : " << width << "  height: " << height << endl;
//        Mat drawImage = Mat::zeros(image.size(), CV_8UC3);
////        imshow("binary image", binary);
//        findContours(src_gray, contours, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
//        for (size_t t = 0; t < contours.size(); t++) {
//            Rect rect = boundingRect(contours[t]);
////        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
//            if (rect.width > (image.cols / 2) && rect.width < (image.cols - 5)) {
//                drawContours(drawImage, contours, static_cast<int>(t), Scalar(0, 0, 255), 2, CV_AA, hireachy, 0,
//                             Point());
//            }
//        }
//        imshow("contours", drawImage);
////        Mat result, bg, fg;
////        Rect rect(srcTri[0].x,srcTri[0].y,srcTri[1].x-srcTri[0].x,srcTri[2].y-srcTri[0].y);//左上坐标（X,Y）和长宽
////        grabCut(image, result, rect, bg, fg, 1, GC_INIT_WITH_RECT);
////        imshow("grab", result);
////        /*threshold(result, result, 2, 255, CV_THRESH_BINARY);
////        imshow("threshold", result);*/
////
////        compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255
////        imshow("result",result);
////        Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
////        image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
////        imshow("foreground", foreground);
////        cv::Mat warpmatrix=cv::getPerspectiveTransform(srcTri,dstTri);
////        cv::warpPerspective(image,resultimg,warpmatrix,Size (width,height),INTER_LINEAR);
//////        cv::warpPerspective(image,resultimg,warpmatrix,resultimg.size(),INTER_LINEAR);
////        cv::imshow("resultimg",resultimg);
////        cv::imwrite("hello.jpg",resultimg);
////        Mat transform=Mat::zeros(3,3,CV_32FC1); //透视变换矩阵
////        transform=getPerspectiveTransform(srcTri,dstTri);  //获取透视变换矩阵
////        warpPerspective(image,imageWarp,transform,Size(image.rows,image.cols));  //透视变换
////        imshow("After WarpPerspecttive",imageWarp);
//    }
//}

/**
 * 90 70
 * 25 374
 * 580 46
 * 688 330
 * @return
 */
int getmat3(int argc, char *argv[]) {
//    cv::Mat src = cv::imread("5.jpg");
//    Mat cdst,dst;
//    cv::imshow("src",src);
//    std::vector<cv::Point2f> src_corners(4);
//    src_corners[0]=cv::Point(90 ,70);
//    src_corners[0]=cv::Point(688, 330);
//    src_corners[0]=cv::Point(580, 46);
//    src_corners[0]=cv::Point(25, 374);
//    std::vector<cv::Point2f> dst_corners(4);
//    src_corners[0]=cv::Point(0 ,0);
//    src_corners[0]=cv::Point(692, 0);
//    src_corners[0]=cv::Point(692, 389);
//    src_corners[0]=cv::Point(0, 389);

//    return  0;
    image = imread("1593681794972.jpg");
    printf("width:%d,height:%d\n", image.cols, image.rows);
    imshow("Source Image", image);
    imageHeight = image.rows;
    imageWidth = image.cols;
    setMouseCallback("Source Image", onMouse);
    waitKey();
    return 0;
}

void show2() {
    VideoCapture cap; //定义视频对象
    cap.open(0); //0为设备的ID号
    while (photo) {
        cap >> image;
        if (!image.empty()) {
            imshow("frame", image);

            imageHeight = image.rows;
            imageWidth = image.cols;
            int linewidt = 90;
            polylines_v.push_back(cv::Point(linewidt, linewidt));
            polylines_v.push_back(cv::Point(linewidt, imageHeight - linewidt));
            polylines_v.push_back(cv::Point(imageWidth - linewidt, imageHeight - linewidt));
            polylines_v.push_back(cv::Point(imageWidth - linewidt, linewidt));
//            resize(frame, frame, Size(640, 480));
            const Point *p = &polylines_v[0];
            int n = (int) polylines_v.size();
            polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3,
                      LINE_AA);
            imshow("polylines", image);
            setMouseCallback("polylines", onMouse);
        }
        waitKey(30); //0.03s获取一帧
    }
    setMouseCallback("Source Image", onMouse);
    waitKey();
}

int main(int argc, char *argv[]) {
//    getMat(argc,argv);
    getmat3(argc,argv);
//    cv::waitKey();
//    show2();
//    show1();
    return 0;
}
