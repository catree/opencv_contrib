// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define DEBUG_DISPLAY 1

namespace opencv_test { namespace {

class CV_GLPMMatcherTest : public cvtest::BaseTest
{
public:
    CV_GLPMMatcherTest(const std::string& datasetName_, int startImg_, int nImgs_, const std::vector<double> eps);
    virtual ~CV_GLPMMatcherTest();

protected:
    virtual void run(int);

    std::string datasetName;
    int startImg;
    int nImgs;
    double correctMatchDistThreshold;
    std::vector<double> epsilon;
};

CV_GLPMMatcherTest::CV_GLPMMatcherTest(const std::string& datasetName_, int startImg_, int nImgs_, const std::vector<double> eps) :
    datasetName(datasetName_), startImg(startImg_), nImgs(nImgs_), correctMatchDistThreshold(5.0), epsilon(eps)
{
}

CV_GLPMMatcherTest::~CV_GLPMMatcherTest() {}

void CV_GLPMMatcherTest::run(int)
{
    ts->set_failed_test_info(cvtest::TS::OK);

    Mat imgRef = imread(string(ts->get_data_path()) + "detectors_descriptors_evaluation/images_datasets/"
                        + datasetName + "/img1.png");
    if (imgRef.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    const bool extended = true;
    Ptr<Feature2D> kaze = KAZE::create(extended);
    vector<KeyPoint> keypointsRef, keypointsCur;
    Mat descriptorsRef, descriptorsCur;

    kaze->detectAndCompute(imgRef, noArray(), keypointsRef, descriptorsRef);

    vector<DMatch> matchesGLPM;

    // Matcher for Lowe ratio test
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    for (int num = startImg; num < startImg+nImgs; num++)
    {
        string imgPath = format("%sdetectors_descriptors_evaluation/images_datasets/%s/img%d.png",
                                ts->get_data_path().c_str(), datasetName.c_str(), num);
        Mat imgCur = imread(imgPath);
        kaze->detectAndCompute(imgCur, noArray(), keypointsCur, descriptorsCur);

        string xml = format("%sdetectors_descriptors_evaluation/images_datasets/%s/H1to%dp.xml",
                            ts->get_data_path().c_str(), datasetName.c_str(), num);
        FileStorage fs(xml, FileStorage::READ);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Mat H1toCur;
        fs[format("H1%d", num)] >> H1toCur;

        TickMeter tm;
        tm.start();
        matchGLPM(keypointsCur, descriptorsCur, keypointsRef, descriptorsRef, matcher, matchesGLPM);
        tm.stop();
        std::cout << "Match " << matchesGLPM.size() << " keypoints from " << keypointsCur.size()
                  << " current keypoints to " << keypointsRef.size() << " reference keypoints in "
                  << tm.getTimeMilli() << " ms" << std::endl;

        int nbCorrectMatches = 0;
        for (size_t i = 0; i < matchesGLPM.size(); i++)
        {
            Point2f ptRef = keypointsRef[matchesGLPM[i].trainIdx].pt;
            Point2f ptCur = keypointsCur[matchesGLPM[i].queryIdx].pt;
            Mat matRef = (Mat_<double>(3,1) << ptRef.x, ptRef.y, 1);
            Mat matTrans = H1toCur * matRef;
            Point2f ptTrans((float) (matTrans.at<double>(0,0)/matTrans.at<double>(2,0)),
                            (float) (matTrans.at<double>(1,0)/matTrans.at<double>(2,0)));

            if (cv::norm(ptTrans-ptCur) < correctMatchDistThreshold)
            {
                nbCorrectMatches++;
            }
        }

        double ratio = nbCorrectMatches / (double) matchesGLPM.size();
        std::cout << "Correct matching ratio (error < " << correctMatchDistThreshold << "): " << ratio << std::endl;

        if (ratio < epsilon[num-startImg])
        {
            ts->printf(cvtest::TS::LOG, "Invalid accuracy for image %s, matching ratio is %f, "
                                        "matching ratio threshold is %f, distance threshold is %f.\n",
                       imgPath.substr(imgPath.size()-8).c_str(), ratio,
                       epsilon[num-startImg], correctMatchDistThreshold);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }

#if DEBUG_DISPLAY
        Mat imgMatchingGLPM;
        drawMatches(imgCur, keypointsCur, imgRef, keypointsRef, matchesGLPM, imgMatchingGLPM, Scalar::all(-1),
                    Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imshow("GLPM Matching", imgMatchingGLPM);
        waitKey();
#endif
    }
}

TEST(XFeatures2d_GLPMMatcher, regression_graf)
{
    std::vector<double> epsilon;
    epsilon.push_back(0.94);
    epsilon.push_back(0.79);
    epsilon.push_back(0.73);
    CV_GLPMMatcherTest test("graf", 2, 3, epsilon);
    test.safe_run();
}

TEST(XFeatures2d_GLPMMatcher, regression_wall)
{
    std::vector<double> epsilon;
    epsilon.push_back(0.97);
    epsilon.push_back(0.94);
    epsilon.push_back(0.92);
    epsilon.push_back(0.9);
    CV_GLPMMatcherTest test("wall", 2, 4, epsilon);
    test.safe_run();
}

}} // namespace
