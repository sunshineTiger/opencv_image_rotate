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

Point2f srcTri[4], dstTri[4];
int clickTimes = 0;  //在图像上单击次数
Mat imageWarp;
bool over = false;

void show1();

void show2();

void onMouse(int event, int x, int y, int flags, void *utsc);

int main() {
//    show1();
    show2();
    return 1;
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

void show1() {
    image = imread("1593681794972.jpg");
    printf("width:%d,height:%d\n", image.cols, image.rows);
    imshow("Source Image", image);
    imageHeight = image.rows;
    imageWidth = image.cols;
    setMouseCallback("Source Image", onMouse);
    waitKey();

}

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

    //sharpen(提高对比度)
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    //make it more sharp
    Mat imgLaplance;
    Mat sharpenImg = src;
    //拉普拉斯算子实现边缘提取
    filter2D(src, imgLaplance, CV_32F, kernel, Point(-1, -1), 0, BORDER_DEFAULT);//拉普拉斯有浮点数计算，位数要提高到32
    src.convertTo(sharpenImg, CV_32F);

    //原图减边缘（白色）实现边缘增强
    Mat resultImg = sharpenImg - imgLaplance;

    resultImg.convertTo(resultImg, CV_8UC3);
    imgLaplance.convertTo(imgLaplance, CV_8UC3);
    imshow("sharpen Image", resultImg);

    //转换成二值图
    Mat binary;

    cvtColor(resultImg, resultImg, CV_BGR2GRAY);

    threshold(resultImg, binary, 40, 255, THRESH_BINARY | THRESH_OTSU);

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
    cvtColor(dst,gray_out,COLOR_BGR2GRAY);
    imshow("gray_out Result", gray_out);
    Mat out_binary;
    threshold(gray_out, out_binary, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);
    imshow("out_binary Result", out_binary);
    kernel = getStructuringElement(MORPH_RECT,Point(5,5));
    morphologyEx(out_binary,out_binary,MORPH_CLOSE,kernel);
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
        Mat caijian_image;
        Mat result, bg, fg;
        int line_width = 100;
        caijian_image = Mat(image,
                            Rect(line_width, line_width, imageWidth - line_width * 2, imageHeight - line_width * 2));
        imshow("caijian_image", caijian_image);

//        Rect rect(line_width, line_width,imageWidth - line_width * 2, imageHeight - line_width * 2);//左上坐标（X,Y）和长宽
        Rect rect(5, 5, caijian_image.cols - 5 * 2, caijian_image.rows - 5 * 2);//左上坐标（X,Y）和长宽
        grabCut(caijian_image, result, rect, bg, fg, 6, GC_INIT_WITH_RECT);
        compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255
        imshow("result", result);
        Mat foreground(caijian_image.size(), CV_8UC3, Scalar(0, 0, 0));
        caijian_image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
        imshow("foreground", foreground);
//        showImage(foreground);
//        Mat src_gray;
//        cvtColor(foreground, src_gray, COLOR_BGR2GRAY);
//        Mat Blur_image;
//        medianBlur(src_gray, Blur_image, 5);
//        imshow("Blur_image", Blur_image);
//        Mat binary;
//
//        threshold(Blur_image, binary, 0, 255, CV_THRESH_BINARY_INV | THRESH_OTSU);
//
//
//        bitwise_not(binary, binary, Mat());
//        imshow("binary", binary);
//        Mat structureElement = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
//        morphologyEx(binary, binary, MORPH_OPEN, structureElement, Point(-1, -1), 1);
//
////        dilate(binary,binary,structureElement,Point(-1,-1), 2);
//        imshow("dilate", binary);
//        Mat canny_image;
//        Canny(binary, canny_image, 200, 400, 3, true);
//        imshow("canny_image", canny_image);
        Mat x_grad, y_grad;
        Mat out_image;
        Sobel(foreground, x_grad, CV_16S, 1, 0, 3);
        Sobel(foreground, y_grad, CV_16S, 0, 1, 3);
        convertScaleAbs(x_grad, x_grad);
        convertScaleAbs(y_grad, y_grad);
        imshow("x_grad", x_grad);
        imshow("y_grad", y_grad);
        addWeighted(x_grad, 0.5, y_grad, 0.5, 0, out_image);
        imshow("addWeightedxx", out_image);
        showImage(foreground);
//        showImage(out_image);
//        vector<vector<Point>> contours;
//        vector<Vec4i> hireachy;
//        Mat drawImage = Mat::zeros(image.size(), CV_8UC3);
//        findContours(binary, contours, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
//        for (size_t t = 0; t < contours.size(); t++) {
//            Rect rect = boundingRect(contours[t]);
////        printf("rect.width : %d, src.cols %d \n ", rect.width, src_image.cols);
//            if (rect.width > (image.cols / 2) && rect.width < (image.cols - 5)) {
//                drawContours(drawImage, contours, static_cast<int>(t), Scalar(0, 0, 255), 2, LINE_AA, hireachy, 0,
//                             Point());
//            }
//        }
//        imshow("contours", drawImage);
    }
}