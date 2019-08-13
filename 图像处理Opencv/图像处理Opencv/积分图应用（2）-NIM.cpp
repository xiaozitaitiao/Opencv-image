#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace  cv;
/*
I:input image
ds:radius of neighborhood window
Ds;radius of search window
a;smoothing parameter of gaussian function
*/
Mat &&NLMfilter(Mat &I, int ds, int Ds, int a)
{
	if (I.channels!=1)
	{
		cout << "I`channel must be 1" << endl;
		exit(0);
	}
	Mat src;
	I.convertTo(src, CV_32FC1);
	src = copyMakeBorder(src,src,);
	int rows = I.rows;
	int cols = I.cols;




}