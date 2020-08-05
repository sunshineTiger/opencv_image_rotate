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
Point2f srcTri[4],dstTri[4];
int clickTimes=0;  //在图像上单击次数
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


void show_Image(Mat src){
    dstTri[2].x=0;
    dstTri[2].y=0;
    dstTri[3].x=500;
    dstTri[3].y=0;
    dstTri[1].x=0;
    dstTri[1].y=500;
    dstTri[0].x=500;
    dstTri[0].y=500;
    cv::Mat resultimg;
    cv::Mat warpmatrix=cv::getPerspectiveTransform(srcTri,dstTri);
    cv::warpPerspective(src,resultimg,warpmatrix,Size (500,500),INTER_LINEAR);
    imshow("warpPerspective",resultimg);
}
int findLargestSquare(const vector<vector<cv::Point> >& squares, vector<cv::Point>& biggest_square)

{

    if (!squares.size()) return -1;

    int max_width = 0;

    int max_height = 0;

    int max_square_idx = 0;

    for (int i = 0; i < squares.size(); i++)

    {

        cv::Rect rectangle = boundingRect(Mat(squares[i]));

        if ((rectangle.width >= max_width) && (rectangle.height >= max_height))

        {

            max_width = rectangle.width;

            max_height = rectangle.height;

            max_square_idx = i;

        }

    }

    biggest_square = squares[max_square_idx];

    return max_square_idx;

}
//检测矩形
//第一个参数是传入的原始图像，第二是输出的图像。
void findSquares(const Mat& image,Mat &out)
{
    int thresh = 50, N = 5;
    vector<vector<Point> > squares;
    squares.clear();

    Mat src,dst, gray_one, gray;

    src = image.clone();
    out = image.clone();
    gray_one = Mat(src.size(), CV_8U);
    //滤波增强边缘检测
    medianBlur(src, dst, 9);
    //bilateralFilter(src, dst, 25, 25 * 2, 35);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    //在图像的每个颜色通道中查找矩形
    for (int c = 0; c < image.channels(); c++)
    {
        int ch[] = { c, 0 };

        //通道分离
        mixChannels(&dst, 1, &gray_one, 1, ch, 1);

        // 尝试几个阈值
        for (int l = 0; l < N; l++)
        {
            // 用canny()提取边缘
            if (l == 0)
            {
                //检测边缘
                Canny(gray_one, gray, 5, thresh, 5);
                //膨脹
                dilate(gray, gray, Mat(), Point(-1, -1));
                imshow("dilate", gray);
            }
            else
            {
                gray = gray_one >= (l + 1) * 255 / N;
            }

            // 轮廓查找
            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            findContours(gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // 检测所找到的轮廓
            for (size_t i = 0; i < contours.size(); i++)
            {
                //使用图像轮廓点进行多边形拟合
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                //计算轮廓面积后，得到矩形4个顶点
                if (approx.size() == 4 &&fabs(contourArea(Mat(approx))) > 1000 &&isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        // 求轮廓边缘之间角度的最大余弦
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    if (maxCosine < 0.3)
                    {
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
    vector<cv::Point> largest_square;
    int idex = findLargestSquare(squares, largest_square);
    for (size_t i = 0; i < squares.size(); i++)
    {
        const Point* p = &squares[i][0];
//        srcTri[clickTimes].x=squares[idex][i].x;
//        srcTri[clickTimes].y=squares[idex][i].y;
//        Point2i newp(p->x,p->y);
//        clickTimes++;
        int n = (int)squares[i].size();
//        circle(out,squares[idex][i],5,Scalar(0,0,255), 5);
//        printf("x:%d y:%d\n",squares[idex][i].x,squares[idex][i].y);
        if (p->x > 3 && p->y > 3)
        {
            polylines(out, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
        }
    }
    imshow("dst",out);
//    clickTimes=0;
//show_Image(out);
waitKey();
}


int main(){
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
    frame=imread("1.jpg");
    findSquares(frame,out);
    return  0;
}

