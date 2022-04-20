#include "KeypointMatcher.h"

void matchBySIFT(Mat img1, Mat img2, int detector, string outFile)
{
	Mat img1Gray, img2Gray;
	if (img1.type() == CV_8UC3)
		cvtColor(img1, img1Gray, COLOR_BGR2GRAY);
	else if (img1.type() == CV_8UC1)
		img1Gray = img1;
	if (img2.type() == CV_8UC3)
		cvtColor(img2, img2Gray, COLOR_BGR2GRAY);
	else if (img2.type() == CV_8UC1)
		img2Gray = img2;

	// detect keypoints for 2 images using SIFT
	vector<CustomKeypoint> queryKeypoints = detectBySift(img1Gray, detector, 10);
	vector<CustomKeypoint> trainKeypoints = detectBySift(img2Gray, detector, 10);
	
	cout << "Collecting descriptors" << endl;

	Mat descriptors1 = collectDescriptors(queryKeypoints);
	Mat descriptors2 = collectDescriptors(trainKeypoints);

	cout << "Converting keypoints" << endl;

	vector<KeyPoint> keypoints1 = convertKeypoints(queryKeypoints);
	vector<KeyPoint> keypoints2 = convertKeypoints(trainKeypoints);

	cout << "Applying KNN" << endl;

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	cout << "Total matches: " << knn_matches.size() << endl;

	// Lowe's test
	const float ratio_thresh = 0.65f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	cout << "Good matches: " << good_matches.size() << endl;

	Mat img_matches;
	drawMatches(img1, keypoints1, img2,	keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("Keypoints Matching", img_matches);
	if (outFile.length() != 0)
		imwrite(outFile, img_matches);
	waitKey(0);
}

// collect descriptors from keypoints to use in drawMatches function
Mat collectDescriptors(vector<CustomKeypoint>& keypoints)
{
	Mat descriptors = Mat::zeros(keypoints.size(), keypoints[0].descriptor.size(), CV_32FC1);
	for (int i = 0; i < keypoints.size(); ++i)
		for (int j = 0; j < keypoints[i].descriptor.size(); ++j)
		{
			descriptors.at<float>(i, j) = saturate_cast<float>((float)keypoints[i].descriptor[j]);
		}
	return descriptors;
}

// convert CustomKeypoint to KeyPoint
vector<KeyPoint> convertKeypoints(vector<CustomKeypoint>& keypoints)
{
	vector<KeyPoint> result;
	for (int i = 0; i < keypoints.size(); ++i)
	{
		KeyPoint newKp = KeyPoint(keypoints[i].pt, keypoints[i].size, keypoints[i].angle, keypoints[i].response, keypoints[i].trueOctave);
		result.push_back(newKp);
	}
	return result;
}
