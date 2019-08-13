#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
void fastNLM(const Mat &input_img, Mat& output_img, int ds, int Ds, int p)
{
	/*
	ds:template window size
	Ds:search window size
	p:Gasuuian parameter
	*/
	int rows = input_img.rows;
	int cols = input_img.cols;
	int pixls = pow(2 * ds + 1, 2);//模板窗口内所有像素量
	Mat src, paddedV;
	copyMakeBorder(input_img, src, Ds + ds, Ds + ds, Ds + ds, Ds + ds, BORDER_REPLICATE);
	copyMakeBorder(input_img, paddedV, Ds, Ds, Ds, Ds, BORDER_REPLICATE);
	paddedV.convertTo(paddedV, CV_64FC1);
	src.convertTo(src, CV_64FC1);
	Mat average = Mat::zeros(rows, cols, CV_64FC1);
	Mat Weight = Mat::zeros(rows, cols, CV_64FC1);
	for (int t1 = -Ds; t1 <= Ds; t1++)
	{
		for (int t2 = -Ds; t2 <= Ds; t2++)
		{
			Mat diff = src(Range(Ds + t1, rows + Ds + 2 * ds + t1), Range(Ds + t2, cols + Ds + 2 * ds + t2)) - src(Range(Ds, rows + Ds + 2 * ds), Range(Ds, cols + Ds + 2 * ds));
			pow(diff, 2, diff);
			Mat diffIntegral;
			integral(diff, diffIntegral, CV_64FC1);
			Mat V = paddedV(Range(Ds + t1, rows + Ds + t1), Range(Ds + t2, cols + Ds + t2));
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					Point p1(i + 2 * ds + 1, j + 2 * ds + 1);
					Point p2(i, j);
					double dist = diffIntegral.at<double>(p1.x, p1.y) + diffIntegral.at<double>(p2.x, p2.y) - diffIntegral.at<double>(p1.x, p2.y) - diffIntegral.at<double>(p2.x, p1.y);
					dist = exp(-dist / (pixls*p));
					Weight.at<double>(i, j) += dist;
					average.at<double>(i, j) += dist * V.at<double>(i, j);
				}
			}



		}
	}
	output_img = average.mul(1 / Weight);
	convertScaleAbs(output_img, output_img);
}


int main()
{
	Mat src = imread("C:/Users/11094/Desktop/美颜2.jpg", 0);
	Mat result1;
	Mat result2;
	fastNlMeansDenoising(src, result1);
	fastNLM(src, result2, 7, 21, 15);
	return 0;

}