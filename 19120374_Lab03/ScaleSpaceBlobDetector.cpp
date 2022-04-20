#include "ScaleSpaceBlobDetector.h"

//check if a point is a blob
bool isBlob(vector<Mat> scaledLaplacian, int x, int y, float current)
{
	for (int i = 0; i < scaledLaplacian.size(); ++i)
	{
		float currentScaleMax = findMax(scaledLaplacian[i], x, y);
		if (currentScaleMax > current) return false;
	}
	return true;
}

float findMax(Mat& scaled, int posX, int posY)
{
	float max = scaled.at<float>(posY, posX);
	// scan neighbors of (posX, posY) to find the max value
	for (int y = posY - 1; y < posY + 2; ++y)
		for (int x = posX - 1; x < posX + 2; ++x)
			if (scaled.at<float>(y, x) > max)
				max = scaled.at<float>(y, x);
	return max;
}

float findMin(Mat& scaled, int posX, int posY)
{
	float min = scaled.at<float>(posY, posX);
	// scan neighbors of (posX, posY) to find the min value
	for (int y = posY - 1; y < posY + 2; ++y)
		for (int x = posX - 1; x < posX + 2; ++x)
			if (scaled.at<float>(y, x) < min)
				min = scaled.at<float>(y, x);
	return min;
}

// normalize values of layers
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

vector<Mat> calculateLoGLayers(Mat& img, int numberOfScales, double initSigma)
{
	vector<Mat> scaledLayers;
	Mat img32;
	img.convertTo(img32, CV_32F);

	for (int i = 1; i <= numberOfScales; ++i)
	{
		Mat filter;
		double sigma = initSigma * pow(sqrt(2), i);
		filter = pow(sigma, 2) * generateLoGFilter(2 * ceil(3 * sigma) + 1, sigma);
		Mat temp;
		filter2D(img32, temp, -1, filter);
		// add borders to make it easy to scan for local maxima
		copyMakeBorder(temp, temp, 1, 1, 1, 1, BORDER_REPLICATE);
		scaledLayers.push_back(temp);
	}
	return scaledLayers;
}

vector<Mat> calculateGaussianImages(Mat& img, int numberOfScales, double initSigma)
{
	vector<Mat> scaledLayers;
	Mat img32;
	img.convertTo(img32, CV_32F);
	
	++numberOfScales;

	for (int i = 1; i <= numberOfScales; ++i)
	{
		Mat filter;
		double sigma = initSigma * pow(sqrt(2), i);
		filter = generateGaussianFilter(2 * ceil(3 * sigma) + 1, sigma);
		Mat temp;
		filter2D(img32, temp, -1, filter);
		// add borders to make it easy to scan for local maxima
		copyMakeBorder(temp, temp, 1, 1, 1, 1, BORDER_REPLICATE);
		scaledLayers.push_back(temp);
	}
	return scaledLayers;
}

// compute DoG layers from blurred images
vector<Mat> calculateDoGLayers(vector<Mat>& blurred)
{
	vector<Mat> scaledLayers;
	for (int i = 1; i < blurred.size(); ++i)
	{
		scaledLayers.push_back(blurred[i] - blurred[i - 1]);
	}
	return scaledLayers;
}

// compute scale-space
vector<Mat> calculateScaledLayers(Mat& img, int numberOfScales, int filterType, double initSigma)
{
	vector<Mat> scaledLayers;

	if (filterType == BLOB_FILTER::DOG_FILTER)
	{
		scaledLayers = calculateGaussianImages(img, numberOfScales, initSigma);
		scaledLayers = calculateDoGLayers(scaledLayers);
	}
	else if (filterType == BLOB_FILTER::LOG_FILTER)
		scaledLayers = calculateLoGLayers(img, numberOfScales, initSigma);

	return scaledLayers;
}

Mat detectBlob(Mat img, int numberOfScales, int filterType, float blobThreshold, double initSigma)
{
	vector<Mat> scaledLayers = calculateScaledLayers(img, numberOfScales, filterType, initSigma);

	Mat result = detectExtrema(scaledLayers, { img.rows, img.cols }, blobThreshold, initSigma);

	return result;
}

// detect all the extrema using the generated scale-space, returns a matrix of pixels contain sizes of detected blobs
Mat detectExtrema(vector<Mat>& scaledLayers, pair<int, int> imageSize, float blobThreshold, double initSigma)
{
	Mat result = Mat::zeros(imageSize.first, imageSize.second, CV_32F);
	for (int i = 1; i < scaledLayers.size() - 1; ++i)
	{
		float radius = pow(sqrt(2), i + 2) * initSigma;
		cout << "\rScanning radius of " << radius;
		double maxVal, minVal;
		minMaxLoc(scaledLayers[i], &minVal, &maxVal);

		vector<Mat> neighborLayers;
		neighborLayers.push_back(scaledLayers[i - 1]);
		neighborLayers.push_back(scaledLayers[i + 1]);
		neighborLayers.push_back(scaledLayers[i]);

		for (int y = 0; y < imageSize.first; ++y)
		{
			for (int x = 0; x < imageSize.second; ++x)
			{
				if (scaledLayers[i].at<float>(y + 1, x + 1) > blobThreshold * maxVal &&
					isBlob(neighborLayers, x + 1, y + 1, scaledLayers[i].at<float>(y + 1, x + 1)))
				{
					result.at<float>(y, x) = i;
				}
			}
		}
	}
	cout << "\nCompleted!\n";
	return result;
}

// visualize blobs
void drawBlob(Mat src, Mat blobs, string outFile, double initSigma)
{
	for (int y = 0; y < src.rows; ++y)
		for (int x = 0; x < src.cols; ++x)
		{
			if (blobs.at<float>(y, x) > 0)
				circle(src, Point2d(x, y), pow(sqrt(2), blobs.at<float>(y, x) + 2) * initSigma, Scalar(0, 255, 0), 1);
		}
	if (outFile.length() != 0)
		imwrite(outFile, src);
	namedWindow("Scaled Norm Blob Detector", WINDOW_AUTOSIZE);
	imshow("Scaled Norm Blob Detector", src);
	waitKey(0);
}

Mat detectDOG(Mat img, int numberOfScales, float blobThreshold)
{
	return detectBlob(img, numberOfScales, BLOB_FILTER::DOG_FILTER, blobThreshold);
}