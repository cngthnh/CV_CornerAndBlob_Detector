#include "SIFT.h"

// initialize octaves
Octave calculateOctave(Mat& img, int numberOfScales, int filterType)
{
	Octave octave;
	if (filterType == BLOB_FILTER::LOG_FILTER)
	{
		octave.type = BLOB_FILTER::LOG_FILTER;
		octave.scaledLayers = calculateLoGLayers(img, numberOfScales);
		octave.blurredImages = calculateGaussianImages(img, numberOfScales);
	}
	else if (filterType == BLOB_FILTER::DOG_FILTER)
	{
		octave.type = BLOB_FILTER::DOG_FILTER;
		octave.blurredImages = calculateGaussianImages(img, numberOfScales);
		octave.scaledLayers = calculateDoGLayers(octave.blurredImages);
	}
	return octave;
}

vector<CustomKeypoint> detectBySift(Mat& src, int detector, int numberOfScales)
{
	Mat img = src.clone();

	vector<Octave> octaves;
	int numberOfOctaves = round(log(min(img.rows, img.cols)) / log(2) - 1);
	
	// compute octaves
	for (int i = 0; i < numberOfOctaves; ++i)
	{
		Octave octave;
		switch (detector)
		{
		case BLOB_FILTER::LOG_FILTER:
			octave = calculateOctave(img, numberOfScales, BLOB_FILTER::LOG_FILTER);
			break;
		case BLOB_FILTER::DOG_FILTER:
			octave = calculateOctave(img, numberOfScales, BLOB_FILTER::DOG_FILTER);
			break;
		}
		octaves.push_back(octave);

		// Sub-sampling the image
		resize(img, img, Size(), 0.5, 0.5, INTER_NEAREST);
	}

	// get keypoints
	vector<CustomKeypoint> keypoints = getListOfKeypoints(octaves, 1.0, 0.5);

	// post processing to eliminate duplicates
	keypoints = cleanKeypoints(keypoints);

	// create descriptors for each of keypoints
	generateDescriptors(keypoints, octaves);

	return keypoints;
}

vector<CustomKeypoint> getListOfKeypoints(vector<Octave> octaves, double initSigma, double contrastThreshold)
{
	vector<CustomKeypoint> result;
	for (int octave = 0; octave < octaves.size(); ++octave)
	{
		pair<int, int> imageSize = { octaves[octave].scaledLayers[0].rows - 2, octaves[octave].scaledLayers[0].cols - 2 };

		// loop through octaves except the first and the last one
		for (int i = 1; i < octaves[octave].scaledLayers.size() - 1; ++i)
		{
			float radius = pow(sqrt(2), i + 2) * initSigma;
			cout << "\rOctave " << octave + 1 << "/" << octaves.size() << " - Scanning radius of " << radius;
			double maxVal, minVal;
			minMaxLoc(octaves[octave].scaledLayers[i], &minVal, &maxVal);

			vector<Mat> neighborLayers;
			neighborLayers.push_back(octaves[octave].scaledLayers[i - 1]);
			neighborLayers.push_back(octaves[octave].scaledLayers[i + 1]);
			neighborLayers.push_back(octaves[octave].scaledLayers[i]);

			for (int y = 0; y < imageSize.first; ++y)
			{
				for (int x = 0; x < imageSize.second; ++x)
				{
					// the coordinates need to be added by 1 because of padding in the DoG layer
					if (isExtrema(neighborLayers, x + 1, y + 1, octaves[octave].scaledLayers[i].at<float>(y + 1, x + 1)))
					{
						// adjust keypoints by quadratic fit
						CustomKeypoint* keypoint = quadraticFitLocalization(y + 1, x + 1, i, octave, octaves, contrastThreshold);
						if (keypoint != NULL)
						{
							vector<CustomKeypoint> keypoints = computeOrientations(*keypoint, octave, octaves[octave].blurredImages[i]);
							for (int i = 0; i < keypoints.size(); ++i)
							{
								result.push_back(keypoints[i]);
							}
							delete keypoint;
						}
					}
				}
			}
		}
	}
	cout << '\n';
	return result;
}

CustomKeypoint* quadraticFitLocalization(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves, double contrastThreshold, int attempts)
{
	// the next to last layer index
	int maxLayerIndex = octaves[0].scaledLayers.size() - 2;
	int imageWidth = octaves[octaveIndex].scaledLayers[0].cols;
	int imageHeight = octaves[octaveIndex].scaledLayers[0].rows;

	Mat extremumUpdate;
	Mat jacobian, hessian;
	int attempt;
	for (attempt = 0; attempt < attempts; ++attempt)
	{
		jacobian = computeJacobianMatrix(y, x, layerIndex, octaveIndex, octaves);
		hessian = computeHessianMatrix(y, x, layerIndex, octaveIndex, octaves);
		solve(hessian, jacobian, extremumUpdate, DECOMP_SVD);
		extremumUpdate = -extremumUpdate;
		if (fabs(extremumUpdate.at<float>(0, 0)) < 0.5
			&& fabs(extremumUpdate.at<float>(1, 0)) < 0.5
			&& fabs(extremumUpdate.at<float>(2, 0)) < 0.5)
			break;
		x += (int) round(extremumUpdate.at<float>(0, 0));
		y += (int) round(extremumUpdate.at<float>(1, 0));
		layerIndex += (int)round(extremumUpdate.at<float>(2, 0));
		if (y<1 || y>imageHeight - 2 || x<1 || x>imageWidth - 2 || layerIndex < 1 || layerIndex > maxLayerIndex)
			// Out of index
			return NULL;
	}

	if (attempt == attempts)
		// Can't converge in "attempts" times
		return NULL;

	float updatedValue = octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x) + 0.5 * Mat(jacobian.t() * extremumUpdate).at<float>(0, 0);

	if (updatedValue < contrastThreshold)
		return NULL;

	// the computed Hessian matrix in the previous step is 3x3 but I just used 2x2 Hessian to compute trace and det
	float trace = hessian.at<float>(0, 0) + hessian.at<float>(1, 1);
	float det = (hessian.at<float>(0, 0) * hessian.at<float>(1, 1)) - (hessian.at<float>(1, 0) * hessian.at<float>(0, 1));

	float r = trace * trace / det;

	if (det > 0 && 10 * pow(trace, 2) < pow(10 + 1, 2) * det)
	{
		CustomKeypoint* keypoint = new CustomKeypoint();
		keypoint->pt = Point2f((x + extremumUpdate.at<float>(0, 0)) * pow(2, octaveIndex), (y + extremumUpdate.at<float>(1, 0)) * pow(2, octaveIndex));
		keypoint->trueOctave = octaveIndex;
		keypoint->octave = octaveIndex + layerIndex * pow(2, 8) + int(round((extremumUpdate.at<float>(2, 0) + 0.5) * 255)) * pow(2, 16);
		keypoint->size = pow(2, ((layerIndex + extremumUpdate.at<float>(2, 0)))) * pow(2, (octaveIndex + 1));
		keypoint->layer = layerIndex;
		keypoint->scale = 1.0 / pow(2, octaveIndex);
		return keypoint;
	}
	return NULL;
}

// calculate jacobian matrix
Mat computeJacobianMatrix(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves)
{
	float dx = 0.5 * (octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x + 1) - octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x - 1));
	float dy = 0.5 * (octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y + 1, x) - octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y - 1, x));
	float ds = 0.5 * (octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y, x) - octaves[octaveIndex].scaledLayers[layerIndex - 1].at<float>(y, x));
	return (Mat_<float>(3, 1) << dx, dy, ds);
}

// calculate hessian matrix
Mat computeHessianMatrix(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves)
{
	float currentValue = octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x);
	float dxx = octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x + 1) - 2 * currentValue + octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y, x - 1);
	float dyy = octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y + 1, x) - 2 * currentValue + octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y - 1, x);
	float dss = octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y, x) - 2 * currentValue + octaves[octaveIndex].scaledLayers[layerIndex - 1].at<float>(y, x);
	float dxy = 0.25 * (octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y + 1, x + 1) + octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y - 1, x - 1) -
		octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y + 1, x - 1) - octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y - 1, x + 1));
	float dxs = 0.25 * (octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y, x + 1) + octaves[octaveIndex].scaledLayers[layerIndex - 1].at<float>(y, x - 1) -
		octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y, x - 1) - octaves[octaveIndex].scaledLayers[layerIndex - 1].at<float>(y, x + 1));
	float dys = 0.25 * (octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y + 1, x) + octaves[octaveIndex].scaledLayers[layerIndex - 1].at<float>(y - 1, x) -
		octaves[octaveIndex].scaledLayers[layerIndex + 1].at<float>(y - 1, x) - octaves[octaveIndex].scaledLayers[layerIndex].at<float>(y + 1, x));
	return (Mat_<float>(3, 3) << 
		dxx, dxy, dxs,
		dxy, dyy, dys,
		dxs, dys, dss);
}

// compute orientations of keypoints by quadratic peak interpolation
vector<CustomKeypoint> computeOrientations(CustomKeypoint keypoint, int octaveIndex, Mat gaussianImage, double radiusFactor, int numberOfHistBins, double peakRatio, double scaleFactor)
{
	vector<CustomKeypoint> orientations;
	int imageWidth = gaussianImage.cols;
	int imageHeight = gaussianImage.rows;
	double scale = scaleFactor * keypoint.size / pow(2, octaveIndex + 1);
	int radius = (int) round(radiusFactor * scale);
	double weightFactor = -0.5 / pow(scale, 2);
	vector<double> rawHist = vector<double>(numberOfHistBins, 0);
	vector<double> smoothedHist = vector<double>(numberOfHistBins, 0);
	for (int i = -radius; i <= radius; ++i)
	{
		// scale to the true octave
		int regionY = (int)round(keypoint.pt.y / pow(2, octaveIndex)) + i;
		if (regionY > 1 && regionY < imageHeight - 2)
		{
			for (int j = -radius; j <= radius; ++j)
			{
				int regionX = (int)round(keypoint.pt.x / pow(2, octaveIndex)) + j;
				if (regionX > 1 && regionX < imageWidth - 2)
				{
					double dx = (double) gaussianImage.at<float>(regionY, regionX + 1) - gaussianImage.at<float>(regionY, regionX - 1);
					double dy = (double) gaussianImage.at<float>(regionY - 1, regionX) - gaussianImage.at<float>(regionY + 1, regionX);
					double gradMag = sqrt(dx * dx + dy * dy);
					// calculate rad and convert it to degree
					double gradOrt = atan2(dy, dx) * (180.0) / M_PI;
					// assign orientation in to bins of histogram
					int histIdx = mod((int)round(gradOrt * numberOfHistBins / 360.0), numberOfHistBins);
					double weight = exp(weightFactor * (pow(i, 2) + pow(j, 2)));
					rawHist[histIdx] += weight * gradMag;
				}
			}
		}
	}

	for (int i = 0; i < numberOfHistBins; ++i)
	{
		smoothedHist[i] = (6 * rawHist[i] + 4 * (rawHist[mod(i-1, numberOfHistBins)] + rawHist[mod(i + 1, numberOfHistBins)]) + rawHist[mod(i - 2, numberOfHistBins)] + rawHist[mod(i + 2, numberOfHistBins)]) / 16.0;
	}

	// find the max value and peak values
	double maxOrientation = *max_element(smoothedHist.begin(), smoothedHist.end());
	vector<int> peaks;
	for (int i = 0; i < smoothedHist.size(); ++i)
	{
		if (smoothedHist[i] > smoothedHist[mod(i - 1, smoothedHist.size())] 
			&& smoothedHist[i] > smoothedHist[mod(i + 1, smoothedHist.size())])
			peaks.push_back(i);
	}

	// smoothen the histogram
	for (int i = 0; i < peaks.size(); ++i)
	{
		if (smoothedHist[peaks[i]] >= peakRatio * maxOrientation)
		{
			double left = smoothedHist[mod(peaks[i] - 1, numberOfHistBins)];
			double right = smoothedHist[mod(peaks[i] + 1, numberOfHistBins)];
			int interpolatedPeakIdx = (int) mod((peaks[i] + 0.5 * (left - right) / (left - 2 * smoothedHist[peaks[i]] + right)), numberOfHistBins);
			double orientation = 360.0 - interpolatedPeakIdx * 360.0 / numberOfHistBins;
			if (fabs(orientation - 360.0) < 0.001)
				orientation = 0;
			keypoint.angle = orientation;
			orientations.push_back(keypoint);
		}
	}
	return orientations;
}

bool compareKeypoints(CustomKeypoint kp1, CustomKeypoint kp2)
{
	if (kp1.pt.x != kp2.pt.x)
		return kp1.pt.x < kp2.pt.x;
	if (kp1.pt.y != kp2.pt.y)
		return kp1.pt.y < kp2.pt.y;
	if (kp1.size != kp2.size)
		return kp2.size < kp1.size;
	if (kp1.angle != kp2.angle)
		return kp1.angle < kp2.angle;
	if (kp1.response != kp2.response)
		return kp2.response < kp1.response;
	if (kp1.octave != kp2.octave)
		return kp2.response < kp1.response;
	return kp2.class_id < kp1.class_id;
}

// drop duplicates
vector<CustomKeypoint> cleanKeypoints(vector<CustomKeypoint>& keypoints)
{
	if (keypoints.size() < 2)
		return keypoints;

	// sort the keypoints and keep unique keypoints
	sort(keypoints.begin(), keypoints.end(), compareKeypoints);

	vector<CustomKeypoint> cleaned = vector<CustomKeypoint>(1, keypoints[0]);

	for (int i = 1; i < keypoints.size(); ++i)
	{
		CustomKeypoint last = cleaned[cleaned.size() - 1];
		if (last.pt.x != keypoints[i].pt.x
			|| last.pt.y != keypoints[i].pt.y
			|| last.size != keypoints[i].size
			|| last.angle != keypoints[i].angle)
			cleaned.push_back(keypoints[i]);
	}

	return cleaned;
}

// generate descriptors for each of keypoints by using trilinear interpolation
void generateDescriptors(vector<CustomKeypoint>& keypoints, vector<Octave>& octaves, int windowSize, int numberOfHistBins, int scaleMultiplier, double maxDescriptorVal)
{
	vector<double> rowBins, colBins, magnitudes, orientationBins;
	for (int i = 0; i < keypoints.size(); ++i)
	{
		rowBins.clear();
		colBins.clear();
		magnitudes.clear();
		orientationBins.clear();
		cout << "\rGenerating descriptors: " << i + 1 << "/" << keypoints.size();
		int octave = keypoints[i].trueOctave, layer = keypoints[i].layer;
		double scale = keypoints[i].scale;
		Mat gaussian = octaves[octave].blurredImages[layer];
		Point2f pt = keypoints[i].pt;
		pt.x *= scale;
		pt.y *= scale;
		double binsPerDegree = numberOfHistBins / 360.0;
		double angle = 360.0 - keypoints[i].angle;
		double rad = angle * M_PI / 180.0;
		double cosAngle = cos(rad), sinAngle = sin(rad);
		double weightMultiplier = -0.5 / pow(0.5 * windowSize, 2);
		vector<vector<vector<double>>> histograms = vector<vector<vector<double>>>(windowSize + 2, vector<vector<double>>(windowSize + 2, vector<double>(numberOfHistBins, 0)));
		double histWidth = scaleMultiplier * 0.5 * scale * keypoints[i].size;
		int halfWidth = (int)round(histWidth * sqrt(2) * (windowSize + 1) * 0.5);
		halfWidth = min(halfWidth, (int) sqrt(gaussian.rows * gaussian.rows + gaussian.cols * gaussian.cols));

		for (int row = -halfWidth; row <= halfWidth; ++row)
			for (int col = -halfWidth; col <= halfWidth; ++col)
			{
				double rowRotation = col * sinAngle + row * cosAngle;
				double colRotation = col * cosAngle - row * sinAngle;
				double rowBin = (rowRotation / histWidth) + 0.5 * windowSize - 0.5;
				double colBin = (colRotation / histWidth) + 0.5 * windowSize - 0.5;
				if (rowBin > -1 && rowBin <windowSize && colBin >-1 and colBin < windowSize)
				{
					int windowRow = round(pt.y + row);
					int windowCol = round(pt.x + col);
					if (windowRow > 1 && windowRow < gaussian.rows - 2 && windowCol > 1 && windowCol < gaussian.cols - 2)
					{
						double dx = gaussian.at<float>(windowRow, windowCol + 1) - gaussian.at<float>(windowRow, windowCol - 1);
						double dy = gaussian.at<float>(windowRow - 1, windowCol) - gaussian.at<float>(windowRow + 1, windowCol);
						double gradMag = sqrt(dx * dx + dy * dy);
						double gradOrt = mod(atan2(dy, dx) * (180.0) / M_PI, 360);
						double weight = exp(weightMultiplier * (pow((rowRotation / histWidth), 2) + pow((colRotation / histWidth), 2)));
						rowBins.push_back(rowBin);
						colBins.push_back(colBin);
						magnitudes.push_back(gradMag * weight);
						orientationBins.push_back((gradOrt - angle) * binsPerDegree);
					}
				}
			}

		// rowBins, colBins, magnitudes and orientationBins are in the same shape now
		for (int k = 0; k < rowBins.size(); ++k)
		{
			int floorRowBin = floor(rowBins[k]);
			int floorColBin = floor(colBins[k]);
			int floorGradOrt = floor(orientationBins[k]);
			double rowFrac = rowBins[k] - floorRowBin;
			double colFrac = colBins[k] - floorColBin;
			double gradOrtFrac = orientationBins[k] - floorGradOrt;

			if (floorGradOrt < 0)
				floorGradOrt += numberOfHistBins;
			if (floorGradOrt >= numberOfHistBins)
				floorGradOrt -= numberOfHistBins;

			double c1, c0, c11, c10, c01, c00, c111, c110, c101, c100, c011, c010, c001, c000;
			c1 = magnitudes[k] * rowFrac;
			c0 = magnitudes[k] * (1 - rowFrac);
			c11 = c1 * colFrac;
			c10 = c1 * (1 - colFrac);
			c01 = c0 * colFrac;
			c00 = c0 * (1 - colFrac);
			c111 = c11 * gradOrtFrac;
			c110 = c11 * (1 - gradOrtFrac);
			c101 = c10 * gradOrtFrac;
			c100 = c10 * (1 - gradOrtFrac);
			c011 = c01 * gradOrtFrac;
			c010 = c01 * (1 - gradOrtFrac);
			c001 = c00 * gradOrtFrac;
			c000 = c00 * (1 - gradOrtFrac);

			histograms[floorRowBin + 1][floorColBin + 1][floorGradOrt] += c000;
			histograms[floorRowBin + 1][floorColBin + 1][mod(floorGradOrt + 1, numberOfHistBins)] += c001;
			histograms[floorRowBin + 1][floorColBin + 2][floorGradOrt] += c010;
			histograms[floorRowBin + 1][floorColBin + 2][mod(floorGradOrt + 1, numberOfHistBins)] += c011;
			histograms[floorRowBin + 2][floorColBin + 1][floorGradOrt] += c100;
			histograms[floorRowBin + 2][floorColBin + 1][mod(floorGradOrt + 1, numberOfHistBins)] += c101;
			histograms[floorRowBin + 2][floorColBin + 2][floorGradOrt] += c110;
			histograms[floorRowBin + 2][floorColBin + 2][mod(floorGradOrt + 1, numberOfHistBins)] += c111;
		}

		vector<double> descriptor = flatten(histograms, 1, -1);
		double threshold = norm(descriptor) * maxDescriptorVal;

		// prevent divide by zero
		double dividend = max(norm(descriptor), 0.001);

		for (int j = 0; j < descriptor.size(); ++j)
		{
			if (descriptor[j] > threshold)
				descriptor[j] = threshold;
			descriptor[j] /= dividend;
			descriptor[j] = saturate_cast<uchar>(round(descriptor[j] * 512));
		}
		keypoints[i].descriptor = descriptor;
	}
	cout << '\n';
}

// check if the current value is a local extremum
bool isExtrema(vector<Mat>& scaledLaplacian, int x, int y, float current)
{
	for (int i = 0; i < scaledLaplacian.size(); ++i)
	{
		if (current < 0)
		{
			float currentScaleMax = findMin(scaledLaplacian[i], x, y);
			if (currentScaleMax < current) return false;
		}
		else
		{
			float currentScaleMax = findMax(scaledLaplacian[i], x, y);
			if (currentScaleMax > current) return false;
		}
	}
	return true;
}