#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        int normType = (descriptorType.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType.compare("MAT_FLANN") == 0) {

        if(descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } else {
        cout << "Unknown matcher type " << matcherType << endl;
        return;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) {
        // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") == 0) {
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        // filter min descriptor distance ratio
        double minDescDistRatio = 0.8;
        for(auto it = knnMatches.begin(); it != knnMatches.end(); ++it) {
            if((*it)[0].distance < (minDescDistRatio * (*it)[1].distance))
                matches.push_back((*it)[0]);
        }

    } else {
        cout << "Unknown selector type " << selectorType << endl;
        return;
    }

    cout << "Number of matched keypints: " << matches.size() << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);

    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::xfeatures2d::SIFT::create();
    } else {
        cout << "Unknown descriptor type " << descriptorType << endl;
        return;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results", bVis);
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    // Apply corner detection
    double t = (double)cv::getTickCount();

    cv::Mat dst, dst_norm; // result matrices for harris
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // detect corners
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // normalize values
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // extract keypoints using non-maximum suppression (NMS)
    double maxOverlap = 0.0;    // max overlap btw. features in %

    // loop over all pixels in the matrix
    for(size_t row = 0; row < dst_norm.rows; ++row) {
        for(size_t col = 0; col < dst_norm.cols; ++col) {

            int response = (int)dst_norm.at<float>(row, col);

            // only responses greater than minResponse will be taken into account
            if(response > minResponse) {
                // create KeyPoint
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(col, row);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform a non-maximum suppression in local neighborhood around key point
                bool bOverlap = false;

                for(auto it = keypoints.begin(); it != keypoints.end(); ++it) {

                    // get the overlap for the new keypoint
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);

                    // check if overlap is above max
                    if(kptOverlap > maxOverlap) {
                        bOverlap = true;

                        // if overlap is over the maximum, take the keypoint with the
                        // best response
                        if(newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                // of the overlap flag is not set, add the new point
                if(bOverlap == false)
                    keypoints.push_back(newKeyPoint);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, "Harris Corner Detector Results", bVis);
}

// Detect keypoints in image using modern detectors FAST, BRISK, ORB, AKAZE or SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // the feature detector pointer
    cv::Ptr<cv::FeatureDetector> detector = nullptr;
    std::string windowName = detectorType + " Keypoint Detector Results";

    if (detectorType.compare("FAST") == 0) {

        // Detector parameters
        int threshold = 30;     // difference between intensity of central px and px of a circle

        // create detector with NMS usage
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, true, type);

    } else if (detectorType.compare("BRISK") == 0) {

        // create detector
        detector = cv::BRISK::create();

    } else if (detectorType.compare("ORB") == 0) {

        // create detector
        detector = cv::ORB::create();

    } else if (detectorType.compare("AKAZE") == 0) {

        // create detector
        detector = cv::AKAZE::create();

    } else if (detectorType.compare("SIFT") == 0) {

        // create detector
        detector = cv::xfeatures2d::SIFT::create();

    } else {
        cout << "Error: Unknown detector type " << detectorType << " configured" << endl;
        return;
    }

    // detect keypoints
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, windowName, bVis);
}

void VisualizeKeypoints(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::string windowName, bool bVis)
{
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
