#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

class WatershedSegmenter {

private:

    cv::Mat markers;

public:

    void setMarkers(const cv::Mat &markerImage) {

        // Convert to image of ints
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(const cv::Mat &image) {

        // Apply watershed
        cv::watershed(image, markers);

        return markers;
    }

    // Return result in the form of an image
    cv::Mat getSegmentation() {

        cv::Mat tmp;
        // all segment with label higher than 255
        // will be assigned value 255
        markers.convertTo(tmp, CV_8U);

        return tmp;
    }

    // Return watershed in the form of an image以图像的形式返回分水岭
    cv::Mat getWatersheds() {

        cv::Mat tmp;
        //在变换前，把每个像素p转换为255p+255（在conertTo中实现）
        markers.convertTo(tmp, CV_8U, 255, 255);

        return tmp;
    }
};

int showImage(Mat &src) {
    Mat binary;
    cv::cvtColor(src, binary, COLOR_BGRA2GRAY);
    cv::threshold(binary, binary, 30, 255, THRESH_BINARY_INV);//阈值分割原图的灰度图，获得二值图像
    // Display the binary image
    cv::namedWindow("binary Image1");
    cv::imshow("binary Image1", binary);

    // CLOSE operation
    cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));//5*5正方形，8位uchar型，全1结构元素
    cv::Mat fg1;
    cv::morphologyEx(binary, fg1, cv::MORPH_CLOSE, element5, Point(-1, -1), 1);// 闭运算填充物体内细小空洞、连接邻近物体

    // Display the foreground image
    cv::namedWindow("Foreground Image");
    cv::imshow("Foreground Image", fg1);
    // Identify image pixels without objects

    cv::Mat bg1;
    cv::dilate(binary, bg1, cv::Mat(), cv::Point(-1, -1), 4);//膨胀4次，锚点为结构元素中心点
    cv::threshold(bg1, bg1, 1, 128, cv::THRESH_BINARY_INV);//>=1的像素设置为128（即背景）
    // Display the background image
    cv::namedWindow("Background Image");
    cv::imshow("Background Image", bg1);
    Mat markers1 = fg1 + bg1; //使用Mat类的重载运算符+来合并图像。
    cv::namedWindow("markers Image");
    cv::imshow("markers Image", markers1);
    WatershedSegmenter segmenter1; //实例化一个分水岭分割方法的对象
    segmenter1.setMarkers(markers1);//设置算法的标记图像，使得水淹过程从这组预先定义好的标记像素开始
    segmenter1.process(src);   //传入待分割原图

    // Display segmentation result
    cv::namedWindow("Segmentation1");
    cv::imshow("Segmentation1", segmenter1.getSegmentation());//将修改后的标记图markers转换为可显示的8位灰度图并返回分割结果（白色为前景，灰色为背景，0为边缘）
    waitKey();
    // Display watersheds
    cv::namedWindow("Watersheds1");
    cv::imshow("Watersheds1", segmenter1.getWatersheds());//以图像的形式返回分水岭（分割线条）
    Mat maskimage = segmenter1.getSegmentation();
    cv::threshold(maskimage, maskimage, 250, 1, THRESH_BINARY);
    cv::cvtColor(maskimage, maskimage, COLOR_GRAY2BGR);

    maskimage = src.mul(maskimage);
    cv::namedWindow("maskimage");
    cv::imshow("maskimage", maskimage);
    waitKey();

    // Turn background (0) to white (255)
    int nl = maskimage.rows; // number of lines
    int nc = maskimage.cols * maskimage.channels(); // total number of elements per line

    for (int j = 0; j < nl; j++) {
        uchar *data = maskimage.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            // process each pixel ---------------------
            if (*data == 0) //将背景由黑色改为白色显示
                *data = 255;
            data++;//指针操作：如为uchar型指针则移动1个字节，即移动到下1列
        }
    }
    cv::namedWindow("result");
    cv::imshow("result", maskimage);
    waitKey(0);
    return 0;
}

void SalientRegionDetectionBasedonLC(Mat &src) {
    int HistGram[256] = {0};
    int row = src.rows, col = src.cols;
    int gray[row][col];
    //int Sal_org[row][col];
    int val;
    Mat Sal = Mat::zeros(src.size(), CV_8UC1);
    Point3_<uchar> *p;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            p = src.ptr<Point3_<uchar> >(i, j);
            val = (p->x + (p->y) * 2 + p->z) / 4;
            HistGram[val]++;
            gray[i][j] = val;
        }
    }

    int Dist[256];
    int Y, X;
    int max_gray = 0;
    int min_gray = 1 << 28;
    for (Y = 0; Y < 256; Y++) {
        val = 0;
        for (X = 0; X < 256; X++)
            val += abs(Y - X) * HistGram[X];                //    论文公式（9），灰度的距离只有绝对值，这里其实可以优化速度，但计算量不大，没必要了
        Dist[Y] = val;
        max_gray = max(max_gray, val);
        min_gray = min(min_gray, val);
    }


    for (Y = 0; Y < row; Y++) {
        for (X = 0; X < col; X++) {
            Sal.at<uchar>(Y, X) = (Dist[gray[Y][X]] - min_gray) * 255 / (max_gray - min_gray);        //    计算全图每个像素的显著性
            //Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性

        }
    }
    imshow("LC", Sal);
//    waitKey(0);

}

void SalientRegionDetectionBasedonAC(Mat &src, int MinR2, int MaxR2, int Scale) {
    Mat Lab;
    cvtColor(src, Lab, CV_BGR2Lab);

    int row = src.rows, col = src.cols;
    int Sal_org[row][col];
    memset(Sal_org, 0, sizeof(Sal_org));

    Mat Sal = Mat::zeros(src.size(), CV_8UC1);

    Point3_<uchar> *p;
    Point3_<uchar> *p1;
    int val;
    Mat filter;

    int max_v = 0;
    int min_v = 1 << 28;
    for (int k = 0; k < Scale; k++) {
        int len = (MaxR2 - MinR2) * k / (Scale - 1) + MinR2;
        blur(Lab, filter, Size(len, len));
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                p = Lab.ptr<Point3_<uchar> >(i, j);
                p1 = filter.ptr<Point3_<uchar> >(i, j);
                //cout<<(p->x - p1->x)*(p->x - p1->x)+ (p->y - p1->y)*(p->y-p1->y) + (p->z - p1->z)*(p->z - p1->z) <<" ";

                val = sqrt((p->x - p1->x) * (p->x - p1->x) + (p->y - p1->y) * (p->y - p1->y) +
                           (p->z - p1->z) * (p->z - p1->z));
                Sal_org[i][j] += val;
                if (k == Scale - 1) {
                    max_v = max(max_v, Sal_org[i][j]);
                    min_v = min(min_v, Sal_org[i][j]);
                }
            }
        }
    }

    cout << max_v << " " << min_v << endl;
    int X, Y;
    for (Y = 0; Y < row; Y++) {
        for (X = 0; X < col; X++) {
            Sal.at<uchar>(Y, X) = (Sal_org[Y][X] - min_v) * 255 / (max_v - min_v);        //    计算全图每个像素的显著性
            //Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        }
    }
    imshow("AC", Sal);
//    waitKey(0);
}

void SalientRegionDetectionBasedonFT(Mat &src) {
    Mat Lab;
    cvtColor(src, Lab, CV_BGR2Lab);

    int row = src.rows, col = src.cols;

    int Sal_org[row][col];
    memset(Sal_org, 0, sizeof(Sal_org));

    Point3_<uchar> *p;

    int MeanL = 0, Meana = 0, Meanb = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            p = Lab.ptr<Point3_<uchar> >(i, j);
            MeanL += p->x;
            Meana += p->y;
            Meanb += p->z;
        }
    }
    MeanL /= (row * col);
    Meana /= (row * col);
    Meanb /= (row * col);

    GaussianBlur(Lab, Lab, Size(3, 3), 0, 0);

    Mat Sal = Mat::zeros(src.size(), CV_8UC1);

    int val;

    int max_v = 0;
    int min_v = 1 << 28;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            p = Lab.ptr<Point3_<uchar> >(i, j);
            val = sqrt((MeanL - p->x) * (MeanL - p->x) + (p->y - Meana) * (p->y - Meana) +
                       (p->z - Meanb) * (p->z - Meanb));
            Sal_org[i][j] = val;
            max_v = max(max_v, val);
            min_v = min(min_v, val);
        }
    }

    cout << max_v << " " << min_v << endl;
    int X, Y;
    for (Y = 0; Y < row; Y++) {
        for (X = 0; X < col; X++) {
            Sal.at<uchar>(Y, X) = (Sal_org[Y][X] - min_v) * 255 / (max_v - min_v);        //    计算全图每个像素的显著性
            //Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        }
    }
    imshow("FT", Sal);
//    waitKey(0);
}

void showImage11(Mat &src) {
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
            if (src.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
                src.at<Vec3b>(x, y)[0] = 0;
                src.at<Vec3b>(x, y)[1] = 0;
                src.at<Vec3b>(x, y)[2] = 0;
            }
        }
    }
    // Show output image
    imshow("Black Background Image", src);
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3, 3) <<
                                    1, 1, 1,
            1, -8, 1,
            1, 1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("New Sharped Image", imgResult);
    src = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);
    // Draw the background marker
    circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
    imshow("Markers", markers * 10000);
    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i, j) = colors[index - 1];
            else
                dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }
    // Visualize the final image
    imshow("Final Result", dst);
}

void showImage12(Mat &src) {
    Mat x_grad, y_grad;
    Mat out_image;
    Sobel(src, x_grad, CV_16S, 1, 0, 3);
    Sobel(src, y_grad, CV_16S, 0, 1, 3);
    convertScaleAbs(x_grad, x_grad);
    convertScaleAbs(y_grad, y_grad);
    imshow("x_grad", x_grad);
    imshow("y_grad", y_grad);
    addWeighted(x_grad, 0.5, y_grad, 0.5, 0, out_image);
    imshow("addWeightedxx", out_image);
    Mat Blur_image;
        medianBlur(out_image, Blur_image, 3);
    Mat gray_src;
    cvtColor(Blur_image, gray_src, COLOR_BGR2GRAY);
    imshow("gray_src", gray_src);
    Mat threshold_src;
    threshold(gray_src, threshold_src, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);
    imshow("threshold_src", threshold_src);
    Mat structureElement = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    morphologyEx(threshold_src, threshold_src, MORPH_CLOSE, structureElement, Point(-1, -1), 1, LINE_AA);

//        dilate(binary,binary,structureElement,Point(-1,-1), 2);
    imshow("morphologyEx_threshold_src", threshold_src);
    Mat canny_src;
    Mat Canny_src;
    Canny(threshold_src, Canny_src, 150, 450, 3, true);
    imshow("gray)canny_src", Canny_src);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
//    Mat drawImage = Mat::zeros(src.size(), CV_8UC3);
    findContours(Canny_src, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    Mat imageContours = Mat(src.size(), CV_8UC3); //最小外接矩形画布
    Mat imageContours1 = Mat(src.size(), CV_8UC3); //最小外结圆画布
    for (int i = 0; i < contours.size(); i++) {
        //绘制轮廓
        drawContours(imageContours, contours, i, Scalar(0, 0, 255), 1, 8, hierarchy);
        drawContours(imageContours1, contours, i, Scalar(255, 0, 0), 1, 8, hierarchy);


        //绘制轮廓的最小外结矩形
        RotatedRect rect = minAreaRect(contours[i]);

        Point2f P[4];
        rect.points(P);
        for (int j = 0; j <= 3; j++) {
            if (rect.boundingRect().width > src.cols /3) {
                line(imageContours, P[j], P[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }

        }
    }

//        //绘制轮廓的最小外结圆
//        Point2f center; float radius;
//        minEnclosingCircle(contours[i],center,radius);
//        circle(imageContours1,center,radius,Scalar(255),2);


    imshow("MinAreaRect", imageContours);
    imshow("MinAreaCircle", imageContours1);
//    imshow("contours", drawImage);
}

int main(int argc, char **argv) {
    Mat src_image;
//    src_image = imread("1.jpg");
//    src_image = imread("933.jpg");
//    src_image = imread("2.jpg");
//    src_image = imread("1127.jpg");
//    src_image = imread("1593422907134.jpg");
//    src_image = imread("1593423129890.jpg");
//    src_image = imread("1593421145441.jpg");
//    src_image = imread("IMG_20200628_173209.jpg");
//    src_image = imread("1593423448512.jpg");
//    src_image = imread("1593681794972.jpg");
    src_image = imread("1593423011560.jpg");
    imshow("原图", src_image);
//    showImage(src_image);
//    SalientRegionDetectionBasedonLC(src_image);
//    SalientRegionDetectionBasedonAC(src_image,10,100,2);
//    SalientRegionDetectionBasedonFT(src_image);
    showImage12(src_image);
    waitKey(0);
    return 0;
}