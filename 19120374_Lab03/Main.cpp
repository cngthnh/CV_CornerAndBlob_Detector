#include "HarrisCornerDetector.h"
#include "ScaleSpaceBlobDetector.h"
#include "SIFT.h"
#include "KeypointMatcher.h"

#define CMD_HARRIS "1"
#define CMD_SCALED_NORM_BLOB_LOG "2"
#define CMD_DOG_BLOB "3"
#define CMD_SIFT "4"
#define CMD_OUTPUT "-o"

int main(int argc, char** argv)
{
	if (argc < 2) {
		cout << "Feature extractor" << '\n';
		return -1;
	}

	string fileName(argv[1]), outFile;

	// Tạo ma trận chứa các giá trị điểm ảnh
	Mat src, image;
	// Đọc file ảnh và giữ nguyên số lượng channels
	src = imread(fileName, IMREAD_UNCHANGED);

	if (!src.data)
	{
		cout << "Image file is inaccessible\n";
		return -1;
	}

	Mat output;
	if (src.type() == CV_8UC3)
		cvtColor(src, image, COLOR_BGR2GRAY);
	else if (src.type() == CV_8UC1)
		image = src;

	// Hiển thị ảnh gốc
	namedWindow("Original Image: " + fileName, WINDOW_AUTOSIZE);
	imshow("Original Image: " + fileName, src);

	if (cmdOptionExists(argv, argc, CMD_OUTPUT))
	{
		char* out = getCmdOption(argv, argc, CMD_OUTPUT);
		outFile = string(out);
	}

	if (strcmp(argv[2], CMD_HARRIS) == 0)
	{
		Mat features = detectHarris(image, 0.04, 0.07);
		showCorners(src, features, outFile);
	}
	else if (strcmp(argv[2], CMD_SCALED_NORM_BLOB_LOG) == 0)
	{
		Mat result = detectBlob(image, 10, LOG_FILTER);
		drawBlob(src, result, outFile);
	}
	else if (strcmp(argv[2], CMD_DOG_BLOB) == 0)
	{
		Mat result = detectDOG(image, 10);
		drawBlob(src, result, outFile);
	}
	else if (strcmp(argv[2], CMD_SIFT) == 0)
	{
		Mat test, testGray;
		string testPath(argv[3]);
		test = imread(testPath, IMREAD_UNCHANGED);
		matchBySIFT(test, src, LOG_FILTER, outFile);
	}
	return 0;
}