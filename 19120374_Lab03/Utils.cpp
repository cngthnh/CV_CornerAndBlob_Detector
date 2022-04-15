#include "Utils.h"

// Hàm lấy các option từ câu lệnh
char* getCmdOption(char** argv, int argc, const char* option)
{
	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], option) == 0) {
			if (++i < argc) {
				return argv[i];
			}
			else return NULL;
		}
	}
	return NULL;
}

// Hàm kiểm tra option có tồn tại trong lời gọi chương trình hay không
bool cmdOptionExists(char** argv, int argc, const char* option)
{
	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], option) == 0) return true;
	}
	return false;
}

// Hàm tính toán hướng gradient và làm tròn về các mốc 0, 45, 90, 135
vector<vector<uchar>> calcDirection(Mat xGrad, Mat yGrad)
{
	vector<vector<uchar>> result;
	if (xGrad.size() != yGrad.size())
	{
		return result;
	}
	for (int y = 0; y < xGrad.rows; ++y)
	{
		vector<uchar> row;
		for (int x = 0; x < xGrad.cols; ++x)
		{
			uchar group;
			double degree = atan(((double)yGrad.at<uchar>(y, x)) / xGrad.at<uchar>(y, x)) * (180.0 / M_PI);
			if ((degree >= -22.5 && degree <= 22.5) || (degree >= 157.5) || (degree <= -157.5))
			{
				group = 0;
			}
			else if ((degree >= -112.5 && degree <= -67.5) || (degree >= 67.5 && degree <= 112.5)) {
				group = 90;
			}
			else if ((degree > 22.5 && degree < 67.5) || (degree > -157.5 && degree < -112.5)) {
				group = 45;
			}
			else if ((degree > 112.5 && degree < 157.5) || (degree > -67.5 && degree < -22.5)) {
				group = 135;
			}
			row.push_back(group);
		}
		result.push_back(row);
	}
	return result;
}

// Tạo bộ lọc Gauss với kích thước và sigma truyền vào
Mat generateGaussianFilter(int size, double sigma)
{
	Mat filter = Mat::zeros(size, size, CV_32F);

	double r, s = 2.0 * sigma * sigma;
	double sum = 0.0;
	int absSize = floor(size / 2);

	for (int y = -absSize; y <= absSize; y++) {
		for (int x = -absSize; x <= absSize; x++) {
			r = sqrt(x * x + y * y);
			double entry = (exp(-(r * r) / s)) / (M_PI * s);
			filter.at<float>(y + absSize, x + absSize) = saturate_cast<float>(entry);
			sum += entry;
		}
	}

	// normalize
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			filter.at<float>(i, j) /= sum;
		}
	}

	return filter;
}

Mat generateLoGFilter(int size, double sigma)
{
	Mat filter = Mat::zeros(size, size, CV_32F);

	double r, sqrSigma = 2.0 * sigma * sigma;
	double quadSigma = pow(sigma, 4);
	double sum = 0.0;
	int absSize = floor(size / 2);

	for (int y = -absSize; y <= absSize; y++) {
		for (int x = -absSize; x <= absSize; x++) {
			r = -(x * x + y * y) / sqrSigma;
			double entry = (-1.0 / (M_PI * quadSigma)) * (1 + r) * exp(r);
			filter.at<float>(y + absSize, x + absSize) = saturate_cast<float>(entry);
			sum += entry;
		}
	}

	// normalize
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			filter.at<float>(i, j) /= sum;
		}
	}

	return filter;
}

void fillGreenDot(Mat& image, int posX, int posY)
{
	for (int y = posY - 1; y < posY + 2; ++y) 
	{
		for (int x = posX - 1; x < posX + 2; ++x)
		{
			image.at<Vec3b>(y, x)[0] = 0;
			image.at<Vec3b>(y, x)[1] = 255;
			image.at<Vec3b>(y, x)[2] = 0;
		}
	}
}

void showCorners(Mat src, Mat features)
{
	for (int y = 1; y < src.rows - 1; ++y)
	{
		for (int x = 1; x < src.cols - 1; ++x)
		{
			if (features.at<uchar>(y, x) > 0)
			{
				fillGreenDot(src, x, y);
			}
		}
	}
	imshow("Harris Corner Detector", src);
	waitKey(0);
}

Mat convertAndSuppressNegatives(Mat img)
{
	Mat result = Mat::zeros(img.size(), CV_8UC1);
	double min, max;
	minMaxLoc(img, &min, &max);
	if (min > 0) min = 0;
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
			result.at<uchar>(y, x) = saturate_cast<uchar>((float)(img.at<float>(y, x)) / (max + min) * 255);
	return result;
}