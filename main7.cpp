#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<vector<cv::Point> > contours, squares, hulls;
vector<cv::Point> hull, approx;
vector<cv::Point> largest_square;
Point2f srcTri[4], dstTri[4];


static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}


void sortdisPorint(Point2f dstPoints[4]) {
    Point temp;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3 - i; j++) {
            if (dstPoints[j].x > dstPoints[j + 1].x) {
                temp = dstPoints[j];
                dstPoints[j] = dstPoints[j + 1];
                dstPoints[j + 1] = temp;
            }
        }
    }
    if (dstPoints[0].y > dstPoints[1].y) {
        temp = dstPoints[0];
        dstPoints[0] = dstPoints[1];
        dstPoints[1] = temp;
    }
    if (dstPoints[2].y < dstPoints[3].y) {
        temp = dstPoints[2];
        dstPoints[2] = dstPoints[3];
        dstPoints[3] = temp;
    }
//    printf("-----x:%f ------y:%f\n", dstPoints[0].x, dstPoints[0].y);
//    printf("-----x:%f ------y:%f\n", dstPoints[1].x, dstPoints[1].y);
//    printf("-----x:%f ------y:%f\n", dstPoints[2].x, dstPoints[2].y);
//    printf("-----x:%f ------y:%f\n", dstPoints[3].x, dstPoints[3].y);
}


int findLargestSquare(const vector<vector<cv::Point> > &squares, vector<cv::Point> &biggest_square) {

    if (!squares.size()) return -1;

    int max_width = 0;

    int max_height = 0;

    int max_square_idx = 0;

    for (int i = 0; i < squares.size(); i++) {
        const Point *p = &squares[i][0];
        if (p->x > 3 && p->y > 3) {
            cv::Rect rectangle = boundingRect(Mat(squares[i]));

            if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {

                max_width = rectangle.width;

                max_height = rectangle.height;

                max_square_idx = i;
            }
        }

    }

    biggest_square = squares[max_square_idx];

    return max_square_idx;

}

//检测矩形
//第一个参数是传入的原始图像，第二是输出的图像。
void buildImage(const Mat &image) {
    clock_t start, finish;
    clock_t start1, finish1;
    clock_t start2, finish2;
    start1 = clock();
    int thresh = 50, N = 3;
    vector<vector<Point> > squares;
    squares.clear();
    Mat out;
    Mat src, dst, gray_one, gray;

    src = image.clone();
    out = image.clone();
    gray_one = Mat(src.size(), CV_8U);

    //滤波增强边缘检测
    medianBlur(src, dst, 3);


    //bilateralFilter(src, dst, 25, 25 * 2, 35);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    start = clock();
    //在图像的每个颜色通道中查找矩形
    cout << "图形处理次数：" << image.channels() << endl;
    int max_width = 0;
    int max_height = 0;
    for (int c = 0; c < image.channels(); c++) {
        int ch[] = {c, 0};

        //通道分离
        mixChannels(&dst, 1, &gray_one, 1, ch, 1);

        // 尝试几个阈值
        for (int l = 0; l < N; l++) {

            // 用canny()提取边缘
            if (l == 0) {
                //检测边缘

                Canny(gray_one, gray, 150, 450, 3);

                //膨脹
                dilate(gray, gray, Mat(), Point(-1, -1));

                imshow("dilate", gray);
            } else {
                gray = gray_one >= (l + 1) * 255 / N;
            }
            start2 = clock();
            // 轮廓查找
            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            finish2 = clock();
            cout << "轮廓查找处理时间：" << double(finish2 - start2) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
            vector<Point> approx;

            // 检测所找到的轮廓
            for (size_t i = 0; i < contours.size(); i++) {
                //使用图像轮廓点进行多边形拟合
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

                //计算轮廓面积后，得到矩形4个顶点
                if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 5000 && isContourConvex(Mat(approx))) {
//                    double maxCosine = 0;

//                    for (int j = 2; j < 5; j++) {
//                        // 求轮廓边缘之间角度的最大余弦
//                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                    }
//
//                    if (maxCosine < 0.6) {

//                        squares.push_back(approx);

                    if (approx.empty()) continue;



//                    for (int i = 0; i < approx.size(); i++) {
                        const Point *p = &approx[0];
                        if (p->x > 3 && p->y > 3) {
                            cv::Rect rectangle = boundingRect(Mat(approx));

                            if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {

                                max_width = rectangle.width;

                                max_height = rectangle.height;

                            }
                        }

//                    }

                    largest_square =approx;

//                    }
                }
            }

        }
    }
    finish = clock();
    cout << "图形处理时间：" << double(finish - start) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;

//    int idex = findLargestSquare(squares, largest_square);
    if (largest_square.empty()) {
        std::cerr << "picture is mark failed" << std::endl;
        return;
    }
    const Point *p = &largest_square[0];
    int n = (int) largest_square.size();
    polylines(out, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
//    for (size_t i = 0; i < squares.size(); i++) {
//        const Point *p = &squares[i][0];
//        int n = (int) squares[i].size();
//        if (p->x > 3 && p->y > 3) {
//            polylines(out, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
//        }
//    }
    for (int i = 0; i < 4; i++) {
        srcTri[i].x = largest_square[3 - i].x;
        srcTri[i].y = largest_square[3 - i].y;
        if (largest_square.size() > 0 ) {
            circle(out, largest_square[i], 5, Scalar(0, 0, 255), 5);
//            printf("x:%d y:%d\n", squares[idex][i].x, squares[idex][i].y);
        }
    }
    imshow("dst", out);


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
        cv::warpPerspective(src, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
    } else {
        //hor横图
        cv::warpPerspective(src, resultimg, warpmatrix, Size(width, height), INTER_LINEAR);
    }

    imshow("warpPerspective", resultimg);
    finish1 = clock();
    cout << "渲染一次使用时间：" << double(finish1 - start1) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
}

void show1() {
    Mat frame = imread("2.jpg");
//    Mat frame = imread("1593423195798.jpg");
    imshow("frame", frame);
    buildImage(frame);
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
            buildImage(frame);
        }
        waitKey(30); //0.03s获取一帧
    }
}

int main(int argc, char *argv[]) {
    show1();
//    show2();
    return 0;

}