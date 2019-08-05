#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
mat stdlocal(const mat &i,int radius)
{
	int rows = i.rows;
	int cols = i.cols;
	mat src;
	copymakeborder(i, src, radius, radius, radius, radius, border_replicate);//扩展边界
	src.convertto(src, cv_32fc1);
	mat stdlocal = mat::zeros(rows,cols, cv_32fc1);
	for (int i=radius;i<rows+radius;i++)
	{
		for (int j=radius;j<cols+radius;j++)
		{
			mat imgrio = mat(src, rect(j - radius, i - radius, 2 * radius+1, 2 * radius+1));

			float meanvalue = mean(imgrio).val[0];
			mat diff(imgrio.rows, imgrio.cols, cv_32fc1);
			diff = imgrio - meanvalue;
			diff = diff.mul(diff);
			stdlocal.at<float>(i-radius,j-radius) = (mean(diff).val[0]);


		}
	}
	return stdlocal;
}

//***********利用积分图加速*************************//
mat faststdlocal(const mat &i, int radius)
{
	int n = pow(2 * radius + 1, 2);//邻域内像素点个数
	int rows = i.rows;
	int cols = i.cols;
	mat src;
	copymakeborder(i, src, radius, radius, radius, radius, border_replicate);//扩展边界
	src.convertto(src, cv_32fc1);
	mat stdlocal = mat::zeros(rows, cols, cv_64fc1);
	mat srcintegral,src2integral;
	integral(src, srcintegral, src2integral, cv_64f);//求积分图


	for (int i = radius; i < rows + radius; i++)
	{
		for (int j = radius; j < cols + radius; j++)
		{
			
			point p11(i - radius ,j - radius);
			point p12(i - radius, j + radius+1);
			point p21(i + radius+1, j - radius);
			point p22(i + radius+1, j + radius+1);
			double sumrio_src2 = src2integral.at<double>(p22.x,p22.y) + src2integral.at<double>(p11.x, p11.y) - src2integral.at<double>(p12.x,p12.y) - src2integral.at<double>(p21.x,p21.y) ;
			double sumrio_src = srcintegral.at<double>(p22.x, p22.y) + srcintegral.at<double>(p11.x, p11.y) - srcintegral.at<double>(p12.x, p12.y) - srcintegral.at<double>(p21.x, p21.y);

			stdlocal.at<double>(i - radius, j - radius) = (sumrio_src2 - (sumrio_src * sumrio_src / n)) / n;


		}
	}
	//normalize(stdlocal, stdlocal, 0, 255, norm_minmax);//归一化
	//convertscaleabs(stdlocal, stdlocal);
	return stdlocal;
}

int main()
{
	mat src= imread("c:/users/11094/desktop/美颜1.jpg",0);
	imshow("原图",src);

	double time0 = static_cast<double>(gettickcount());
	mat resut1 = stdlocal(src, 1);
	time0 = ((double)gettickcount() - time0 )/ gettickfrequency();
	cout << "stdlocal() time:" << time0<<endl;

	double time1 = static_cast<double>(gettickcount());
	mat resut2 = faststdlocal(src, 1);
	time1 = ((double)gettickcount() - time1) / gettickfrequency();
	cout << "faststdlocal() time:" << time1<<endl;

	imshow("标准差", resut1);
	imshow("积分图加速-标准差", resut2);
	normalize(resut1, resut1, 0, 255, norm_minmax);
	normalize(resut2, resut2, 0, 255, norm_minmax);
	convertscaleabs(resut1, resut1);
	convertscaleabs(resut2, resut2);
	imshow("归一化-标准差", resut1);
	imshow("积分图加速-归一化标准差", resut2);
	waitkey(0);

	return 0;
}