#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

int thresh = 50, N = 5;
const char *wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat &image, vector<vector<Point> > &squares) {
    squares.clear();

//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());


    // blur will enhance edge detection
    Mat timg(image);
    medianBlur(image, timg, 9);//中值滤波去噪；
    Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; l++) {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0) {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
            } else {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++) {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx))) {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++) {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
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

int idex = -1;

// the function draws all the squares in the image
static void drawSquares(Mat &image, const vector<vector<Point> > &squares) {
    for (size_t i = 0; i < squares.size(); i++) {
        const Point *p = &squares[i][0];
        int n = (int) squares[i].size();
        //dont detect the border
        if (p->x > 3 && p->y > 3) {
            idex = i;

            polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
        }
    }


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
    printf("-----x:%f ------y:%f\n", dstPoints[0].x, dstPoints[0].y);
    printf("-----x:%f ------y:%f\n", dstPoints[1].x, dstPoints[1].y);
    printf("-----x:%f ------y:%f\n", dstPoints[2].x, dstPoints[2].y);
    printf("-----x:%f ------y:%f\n", dstPoints[3].x, dstPoints[3].y);
}

Point2f srcTri[4], dstTri[4];

int main(int /*argc*/, char ** /*argv*/) {
    vector<vector<Point> > squares;
    Mat image = imread("8.png", 1);
    Mat src;
    Mat outPic;
    image.copyTo(src);
    if (image.empty()) {
//            cout << "Couldn't load " << names[i] << endl;
        return -1;
    }

    findSquares(image, squares);
    drawSquares(image, squares);
    vector<cv::Point> largest_square;

    int idex = findLargestSquare(squares, largest_square);
    if (idex <= 0) {
        std::cerr << "picture is mark failed" << std::endl;
        return -1;
    }
    Point2f srcPoints[4], dstPoints[4];
    int p1x = squares[idex][0].x;
    int p1y = squares[idex][0].y;
    int p2x = squares[idex][1].x;
    int p2y = squares[idex][1].y;
    int p3x = squares[idex][3].x;
    int p3y = squares[idex][3].y;
    double dis1 = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
    double dis2 = sqrt((p1x - p3x) * (p1x - p3x) + (p1y - p3y) * (p1y - p3y));
    int width = -1;
    int height = -1;
    if (dis1 > dis2) {
        //ver竖图
        double bili = dis1 / dis2;
        if (bili <= 1.1) {
            dstPoints[0].x = 0;
            dstPoints[0].y = 0;
            dstPoints[1].x = 0;
            dstPoints[1].y = 400;
            dstPoints[2].x = 400;
            dstPoints[2].y = 400;
            dstPoints[3].x = 400;
            dstPoints[3].y = 0;
            sortdisPorint(dstPoints);
            width = 400;
            height = 400;
        } else {
            dstPoints[0].x = 0;
            dstPoints[0].y = 0;
            dstPoints[1].x = 0;
            dstPoints[1].y = (int) (dis2 * bili);
            dstPoints[2].x = (int) dis2;
            dstPoints[2].y = (int) (dis2 * bili);
            dstPoints[3].x = (int) dis2;
            dstPoints[3].y = 0;
            sortdisPorint(dstPoints);
            width = (int) dis2;
            height = (int) (dis2 * bili);
        }
    } else if (dis1 < dis2) {
        //hor横图
        double bili = dis2 / dis1;
        if (bili <= 1.1) {
            dstPoints[0].x = 0;
            dstPoints[0].y = 0;
            dstPoints[1].x = 0;
            dstPoints[1].y = 400;
            dstPoints[2].x = 400;
            dstPoints[2].y = 400;
            dstPoints[3].x = 400;
            dstPoints[3].y = 0;
            sortdisPorint(dstPoints);
            width = 400;
            height = 400;

        } else {
            dstPoints[0].x = 0;
            dstPoints[0].y = 0;
            dstPoints[1].x = 0;
            dstPoints[1].y = (int) dis1;
            dstPoints[2].x = (int) (dis1 * bili);
            dstPoints[2].y = (int) dis1;
            dstPoints[3].x = (int) (dis1 * bili);
            dstPoints[3].y = 0;
            sortdisPorint(dstPoints);
            width = (int) dis2;
            height = (int) (dis2 * bili);
        }
    } else {
        //equ
        dstPoints[0].x = 0;
        dstPoints[0].y = 0;
        dstPoints[1].x = 0;
        dstPoints[1].y = 400;
        dstPoints[2].x = 400;
        dstPoints[2].y = 400;
        dstPoints[3].x = 400;
        dstPoints[3].y = 0;
        sortdisPorint(dstPoints);
        width = 400;
        height = 400;
    }  printf("-----width:%d ------height:%d\n", width, height);
//    int tempx = squares[idex][0].x;
//    int tempy = squares[idex][0].y;
//    for (int j = 1; j < 4; j++) {
//        int compx = squares[idex][j].x;
//        int compy = squares[idex][j].y;
//        if (tempx > compx) {
//            tempx = compx;
//        }
//        if (tempy > compy) {
//            tempy = compx;
//        }
//    }
//    srcPoints[0].x=tempx;
//    srcPoints[0].y=tempx;
    vector<Point> sortsquares;
    sortsquares.push_back(squares[idex][0]);
    sortsquares.push_back(squares[idex][1]);
    sortsquares.push_back(squares[idex][2]);
    sortsquares.push_back(squares[idex][3]);
    Point temp;
    for (int i = 0; i < sortsquares.size(); i++) {
        for (int j = 0; j < sortsquares.size() - 1 - i; j++) {
            if (sortsquares[j].x > sortsquares[j + 1].x) {
                temp = sortsquares[j];
                sortsquares[j] = sortsquares[j + 1];
                sortsquares[j + 1] = temp;
            }
        }
    }
    if (sortsquares[0].y > sortsquares[1].y) {
        temp = sortsquares[0];
        sortsquares[0] = sortsquares[1];
        sortsquares[1] = temp;
    }
    if (sortsquares[2].y < sortsquares[3].y) {
        temp = sortsquares[2];
        sortsquares[2] = sortsquares[3];
        sortsquares[3] = temp;
    }

    for (int i = 0; i < 4; i++) {
        srcPoints[i].x = sortsquares[i].x;
        srcPoints[i].y = sortsquares[i].y;
        printf("-----x:%f ------y:%f\n", srcPoints[i].x, srcPoints[i].y);
        circle(image, srcPoints[i], 5, Scalar(0, 0, 255), 5);
    }
    imshow(wndname, image);
    Mat transMat = getPerspectiveTransform(srcPoints, dstPoints);
    cv::warpPerspective(src, outPic, transMat, Size(width, height), INTER_LINEAR);
    imshow("outPic", outPic);
    waitKey();
    return 0;
}