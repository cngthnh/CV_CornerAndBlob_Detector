#include "HarrisCornerDetector.h"
#include "ScaleSpaceBlobDetector.h"

#define CMD_HARRIS "1"
#define CMD_SCALED_NORM_BLOB_LOG "2"
#define CMD_DOG_BLOB "3"

int main(int argc, char** argv)
{
	if (argc < 2) {
		cout << "Feature extractor" << '\n';
		return -1;
	}

	string fileName(argv[1]);

	// Tạo ma trận chứa các giá trị điểm ảnh
	Mat src, image;
	// Đọc file ảnh và giữ nguyên số lượng channels
	src = imread(fileName, IMREAD_UNCHANGED);

	if (!src.data)
	{
		cout << "Image file is inaccessible\n";
		return -1;
	}

	// Hiển thị ảnh gốc
	namedWindow("Original Image: " + fileName, WINDOW_AUTOSIZE);
	imshow("Original Image: " + fileName, src);

	Mat output;
	if (src.type() == CV_8UC3)
		cvtColor(src, image, COLOR_BGR2GRAY);
	else if (src.type() == CV_8UC1)
		image = src;

	if (strcmp(argv[2], CMD_HARRIS) == 0)
	{
		Mat features = detectHarris(image, 0.04, 0.1);
		showCorners(src, features);
	}
	else if (strcmp(argv[2], CMD_SCALED_NORM_BLOB_LOG) == 0)
	{
		Mat result = detectBlob(image, 10, LOG_FILTER);
		drawBlob(src, result);
	}
	else if (strcmp(argv[2], CMD_DOG_BLOB) == 0)
	{
		Mat result = detectDOG(image, 10);
		drawBlob(src, result);
	}
	return 0;
}