#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
Mat stdLocal(const Mat &I,int radius)
{
	int rows = I.rows;
	int cols = I.cols;
	Mat src;
	copyMakeBorder(I, src, radius, radius, radius, radius, BORDER_REPLICATE);//扩展边界
	src.convertTo(src, CV_32FC1);
	Mat stdLocal = Mat::zeros(rows,cols, CV_32FC1);
	for (int i=radius;i<rows+radius;i++)
	{
		for (int j=radius;j<cols+radius;j++)
		{
			Mat imgRIO = Mat(src, Rect(j - radius, i - radius, 2 * radius+1, 2 * radius+1));

			float meanValue = mean(imgRIO).val[0];
			Mat diff(imgRIO.rows, imgRIO.cols, CV_32FC1);
			diff = imgRIO - meanValue;
			diff = diff.mul(diff);
			stdLocal.at<float>(i-radius,j-radius) = (mean(diff).val[0]);


		}
	}
	return stdLocal;
}

//***********利用积分图加速*************************//
Mat fastStdLocal(const Mat &I, int radius)
{
	int N = pow(2 * radius + 1, 2);//邻域内像素点个数
	int rows = I.rows;
	int cols = I.cols;
	Mat src;
	copyMakeBorder(I, src, radius, radius, radius, radius, BORDER_REPLICATE);//扩展边界
	src.convertTo(src, CV_32FC1);
	Mat stdLocal = Mat::zeros(rows, cols, CV_64FC1);
	Mat srcIntegral,src2Integral;
	integral(src, srcIntegral, src2Integral, CV_64F);//求积分图


	for (int i = radius; i < rows + radius; i++)
	{
		for (int j = radius; j < cols + radius; j++)
		{
			
			Point p11(i - radius ,j - radius);
			Point p12(i - radius, j + radius+1);
			Point p21(i + radius+1, j - radius);
			Point p22(i + radius+1, j + radius+1);
			double sumRIO_src2 = src2Integral.at<double>(p22.x,p22.y) + src2Integral.at<double>(p11.x, p11.y) - src2Integral.at<double>(p12.x,p12.y) - src2Integral.at<double>(p21.x,p21.y) ;
			double sumRIO_src = srcIntegral.at<double>(p22.x, p22.y) + srcIntegral.at<double>(p11.x, p11.y) - srcIntegral.at<double>(p12.x, p12.y) - srcIntegral.at<double>(p21.x, p21.y);

			stdLocal.at<double>(i - radius, j - radius) = (sumRIO_src2 - (sumRIO_src * sumRIO_src / N)) / N;


		}
	}
	//normalize(stdLocal, stdLocal, 0, 255, NORM_MINMAX);//归一化
	//convertScaleAbs(stdLocal, stdLocal);
	return stdLocal;
}

int main()
{
	Mat src= imread("C:/Users/11094/Desktop/美颜1.jpg",0);
	imshow("原图",src);

	double time0 = static_cast<double>(getTickCount());
	Mat resut1 = stdLocal(src, 1);
	time0 = ((double)getTickCount() - time0 )/ getTickFrequency();
	cout << "stdLocal() time:" << time0<<endl;

	double time1 = static_cast<double>(getTickCount());
	Mat resut2 = fastStdLocal(src, 1);
	time1 = ((double)getTickCount() - time1) / getTickFrequency();
	cout << "fastStdLocal() time:" << time1<<endl;

	imshow("标准差", resut1);
	imshow("积分图加速-标准差", resut2);
	normalize(resut1, resut1, 0, 255, NORM_MINMAX);
	normalize(resut2, resut2, 0, 255, NORM_MINMAX);
	convertScaleAbs(resut1, resut1);
	convertScaleAbs(resut2, resut2);
	imshow("归一化-标准差", resut1);
	imshow("积分图加速-归一化标准差", resut2);
	waitKey(0);

	return 0;
}