
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // get all matches with keypoints in the bounding box
    vector<cv::DMatch> roiMatches;
    double meanDistance = 0.F;

    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it) {
        auto keypoint = kptsCurr.at(it->trainIdx);

        if(boundingBox.roi.contains(cv::Point(keypoint.pt.x, keypoint.pt.y))) {
            roiMatches.push_back(*it);
            meanDistance += it->distance;
        }
    }

    // calculate the distance mean for alter outliers filtering
    cout << "Matched keypoints to ROI: " << roiMatches.size() << endl;
    if(roiMatches.size() > 0)
        meanDistance /= roiMatches.size();
    else
        return;

    // only push keypoints below a specific threshold of the mean
    auto thresh = 0.8 * meanDistance;
    for(auto it = roiMatches.begin(); it != roiMatches.end(); ++it)
        if(it->distance < thresh)
            boundingBox.kptMatches.push_back(*it);

    cout << "Lasting keypoints after filtering: " << boundingBox.kptMatches.size() << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // calculate time between two frames
    double dT = 1. / frameRate;
    double laneWidth = 3.75;    // lane width, assumed for German city lanes

    // filter the lidar points to the ego lane
    vector<LidarPoint> lidarFilteredPrev, lidarFilteredCurr;

    // filter points in ego lane
    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
        if(abs(it->y) <= (laneWidth / 2.0))
            lidarFilteredPrev.push_back(*it);

    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
        if(abs(it->y) <= (laneWidth / 2.0))
            lidarFilteredCurr.push_back(*it);

    // try to compensate outliers by taking the median of all measured points
    // --> when averaging, outliers have too much influence on the values
    sort(lidarFilteredPrev.begin(), lidarFilteredPrev.end(), [](LidarPoint a, LidarPoint b)->bool{
            return a.x < b.x;
        });

    sort(lidarFilteredCurr.begin(), lidarFilteredCurr.end(), [](LidarPoint a, LidarPoint b)->bool{
            return a.x < b.x;
        });

    double medianDistXPrev = lidarFilteredPrev[lidarFilteredPrev.size() / 2].x;
    double medianDistXCurr = lidarFilteredCurr[lidarFilteredCurr.size() / 2].x;

    TTC = medianDistXCurr * dT / (medianDistXPrev - medianDistXCurr);
}

static vector<int> getBoundingBoxIDs(const DataFrame& frame, const int kpIndex)
{
    vector<int> ret;
    // get point from keypoint for Rect compare
    auto pt = cv::Point(frame.keypoints.at(kpIndex).pt.x, frame.keypoints.at(kpIndex).pt.y);

    // loop over bounding boxes and check for matches
    for(size_t i = 0; i < frame.boundingBoxes.size(); ++i)
        if(frame.boundingBoxes.at(i).roi.contains(pt))
            ret.push_back(frame.boundingBoxes.at(i).boxID);

    return std::move(ret);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // for each bounding box in the previous frame every bounding box in the new frame is a potential candidate
    // so this may be solved with a two dimensional array of matching counts
    // in scenes with a very very large count of boxes this may waste memory, so associative containers
    // seem to be a better solution
    map<int, map<int, unsigned int>> kptCounts;    // [prevBB][currBB] count

    // loop over all matches to process them
    for(auto matchesIt = matches.begin(); matchesIt != matches.end(); ++matchesIt) {

        // get bounding boxes indices of previous and current frame
        auto prevBBs = getBoundingBoxIDs(prevFrame, matchesIt->queryIdx);
        auto currBBs = getBoundingBoxIDs(currFrame, matchesIt->trainIdx);

        // if we have matches in both frames, fill the map
        if(!prevBBs.empty() && !currBBs.empty()) {

            // loop over previous BBs as key
            for(auto prevBB : prevBBs) {

                // loop over all current BBs as index
                for(auto currBB : currBBs) {

                    // increment or insert based on existing element
                    auto p = kptCounts.find(prevBB);
                    if(p != kptCounts.end()) {

                        // create or increment current BB
                        auto c = p->second.find(currBB);
                        if(c != p->second.end())
                            c->second++;
                        else
                            p->second.insert(pair<size_t, unsigned int>(currBB, 1));
                    } else {

                        // create and insert prev and current BB
                        kptCounts[prevBB][currBB] = 1;
                    }
                }
            }
        }
    }

    // for each previous bounding box, find the best match and put to output vector
    for(auto p : kptCounts) {

        // check current boxes and find the one with highest count
        size_t bestId = 0;
        for(auto c : p.second)
            if(c.second > bestId)
                bestId = c.first;

        // if found best, add to return map
        if(bestId > 0)
            bbBestMatches[p.first] = bestId;
    }

    cout << "Found " << bbBestMatches.size() << " matches: " << endl;
}
