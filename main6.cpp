#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<vector<cv::Point> > contours, squares, hulls;
vector<cv::Point> approx;
vector<cv::Point> largest_square;
Point2f srcTri[4], dstTri[4];
vector<Vec4i> hierarchy;

//检测矩形
//第一个参数是传入的原始图像，第二是输出的图像。
void buildImage(Mat &image) {
    clock_t start, finish;
    clock_t start1, finish1;
    clock_t start2, finish2;
    clock_t start3, finish3;
    clock_t start4, finish4;
    clock_t start5, finish5;
    start1 = clock();

    int thresh = 50, N = 3;
//    vector<vector<Point> > squares;
    squares.clear();
    Mat out;
    Mat src, dst, gray_one, gray;
//    start5 = clock();
    src = image.clone();
    out = image.clone();
    gray_one = Mat(src.size(), CV_8U);
//    finish5 = clock();
//    cout << " 处理时间：" << double(finish5 - start5) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
    //滤波增强边缘检测
    medianBlur(image, dst, 9);


    //bilateralFilter(src, dst, 25, 25 * 2, 35);

//    vector<vector<Point> > contours;

    hierarchy.clear();
    start = clock();
    //在图像的每个颜色通道中查找矩形
//    cout << "图形处理次数：" << image.channels() << endl;
    int max_width = 0;
    int max_height = 0;
    largest_square.clear();
    for (int c = 0; c < image.channels(); c++) {
        int ch[] = {c, 0};

        //通道分离
        mixChannels(&dst, 1, &gray_one, 1, ch, 1);
        string s = "mixChannels";
        ostringstream oss;
        oss << s << c;
//        imshow(oss.str(), gray_one);
        // 尝试几个阈值
        for (int l = 0; l < N; l++) {
            start3 = clock();
            // 用canny()提取边缘
            if (l == 0) {
                //检测边缘
                Canny(gray_one, gray, 150, 450, 3);

                //膨脹
                dilate(gray, gray, Mat(), Point(-1, -1));

                imshow("dilate", gray);
            } else {

                gray = gray_one >= (l + 1) * 255 / N;
//                gray = gray_one >= 100;
//                string s = "dilate";
//                ostringstream oss;
//                oss << s << c << l;
//                imshow(oss.str(), gray);

            }
            finish3 = clock();
            cout << "用canny()提取边缘处理时间：" << double(finish3 - start3) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
            // 轮廓查找
            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            start4 = clock();
            findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            finish4 = clock();
            cout << " 轮廓查找处理时间：" << double(finish4 - start4) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
            approx.clear();
            // 检测所找到的轮廓
            for (size_t i = 0; i < contours.size(); i++) {
                //使用图像轮廓点进行多边形拟合
//                cout<<"xxxxxx:"<<arcLength(Mat(contours[i]), true)<<endl;
                if (arcLength(Mat(contours[i]), true) < 400)
                    continue;
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

                //计算轮廓面积后，得到矩形4个顶点
                if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx))) {
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

                    largest_square = approx;

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

    for (int i = 0; i < 4; i++) {
        srcTri[i].x = largest_square[3 - i].x;
        srcTri[i].y = largest_square[3 - i].y;
        if (!largest_square.empty()) {
            circle(out, largest_square[i], 5, Scalar(0, 0, 255), 5);
//            printf("x:%d y:%d\n", squares[idex][i].x, squares[idex][i].y);
        }
    }
    imshow("dst", out);
    start2 = clock();

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
    finish2 = clock();
    cout << "变换处理时间：" << double(finish2 - start2) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
    imshow("warpPerspective", resultimg);
    finish1 = clock();
    cout << "渲染一次使用时间：" << double(finish1 - start1) / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
}




void show1() {
//    Mat frame = imread("4.png");
    Mat frame = imread("2.jpg");
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
//    show1();
    show2();
    return 0;

}