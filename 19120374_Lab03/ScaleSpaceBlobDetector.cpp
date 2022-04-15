#include "ScaleSpaceBlobDetector.h"

bool isBlob(vector<Mat> scaledLaplacian, int x, int y, float current)
{
	for (int i = 0; i < scaledLaplacian.size(); ++i)
	{
		float currentScaleMax = findMax(scaledLaplacian[i], x, y);
		if (currentScaleMax > current) return false;
	}
	return true;
}

float findMax(Mat scaled, int posX, int posY)
{
	float max = scaled.at<float>(posY, posX);
	// duyệt các điểm kề với điểm (posX, posY) để tìm max
	for (int y = posY - 1; y < posY + 2; ++y)
		for (int x = posX - 1; x < posX + 2; ++x)
			if (scaled.at<float>(y, x) > max)
				max = scaled.at<float>(y, x);
	return max;
}

void normalize(vector<Mat>& imgs)
{
	double overallMin = 255, overallMax = 0;
	for (int i = 0; i < imgs.size(); ++i)
	{
		double minVal, maxVal;
		minMaxLoc(imgs[i], &minVal, &maxVal);
		overallMin = min(overallMin, minVal);
		overallMax = max(overallMax, maxVal);
	}
	float range = overallMax - overallMin;
	for (int i = 0; i < imgs.size(); ++i)
	{
		for (int y = 0; y < imgs[i].rows; ++y)
			for (int x = 0; x < imgs[i].cols; ++x)
			{
				imgs[i].at<float>(y, x) = saturate_cast<float>((float)(imgs[i].at<float>(y, x)) / range * 255);
			}
	}
}

Mat detectBlob(Mat img, int numberOfScales, int filterType, float blobThreshold)
{
	double initSigma = 1.0;
	vector<Mat> scaledLayers;
	Mat result = Mat::zeros(img.size(), CV_32F);
	Mat img32;
	img.convertTo(img32, CV_32F);
	filter2D(img32, img32, -1, generateGaussianFilter(2 * ceil(3 * 1.2) + 1, 1.2));

	if (filterType == BLOB_FILTER::DOG_FILTER)
		++numberOfScales;

	for (int i = 1; i <= numberOfScales; ++i)
	{
		Mat filter;
		double sigma = initSigma * pow(sqrt(2), i);
		if (filterType == BLOB_FILTER::LOG_FILTER)
			filter = pow(sigma, 2) * generateLoGFilter(2 * ceil(3 * sigma) + 1, sigma);
		else
			filter = generateGaussianFilter(2 * ceil(3 * sigma) + 1, sigma);
		Mat temp;
		filter2D(img32, temp, -1, filter);
		// thêm border để dễ scan local maxima
		copyMakeBorder(temp, temp, 1, 1, 1, 1, BORDER_REPLICATE);
		scaledLayers.push_back(temp);
	}

	if (filterType == BLOB_FILTER::DOG_FILTER)
	{
		for (int i = scaledLayers.size() - 1; i > 0; --i)
		{
			scaledLayers[i] -= scaledLayers[i - 1];
		}
		scaledLayers.erase(scaledLayers.begin());
	}

	for (int i = 1; i < scaledLayers.size() - 1; ++i)
	{
		float radius = pow(sqrt(2), i + 2) * initSigma;
		cout << "Scanning radius of " << radius << '\n';
		double maxVal, minVal;
		minMaxLoc(scaledLayers[i], &minVal, &maxVal);

		vector<Mat> neighborLayers;
		neighborLayers.push_back(scaledLayers[i - 1]);
		neighborLayers.push_back(scaledLayers[i + 1]);
		neighborLayers.push_back(scaledLayers[i]);

		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				if (isBlob(neighborLayers, x + 1, y + 1, scaledLayers[i].at<float>(y + 1, x + 1)) && scaledLayers[i].at<float>(y + 1, x + 1) > blobThreshold * maxVal)
				{
					result.at<float>(y, x) = radius;
				}
			}
		}
	}
	return result;
}

void drawBlob(Mat src, Mat blobs)
{
	for (int y = 0; y < src.rows; ++y)
		for (int x = 0; x < src.cols; ++x)
		{
			if (blobs.at<float>(y, x) > 0)
				circle(src, Point2d(x, y), blobs.at<float>(y, x), Scalar(0, 255, 0), 1);
		}
	namedWindow("Scaled Norm Blob Detector", WINDOW_AUTOSIZE);
	imshow("Scaled Norm Blob Detector", src);
	waitKey(0);
}

Mat detectDOG(Mat img, int numberOfScales, float blobThreshold)
{
	return detectBlob(img, numberOfScales, BLOB_FILTER::DOG_FILTER, blobThreshold);
}