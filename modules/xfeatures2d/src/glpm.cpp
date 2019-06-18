// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is based on file issued with the following license:

/*============================================================================

Copyright <2019> <Jiayi Ma, Junjun Jiang, Huabing Zhou, Ji Zhao, and Xiaojie Guo>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "precomp.hpp"
#include <map>

namespace cv
{
namespace xfeatures2d
{
static void nearestNeighbor(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
                            int K, std::vector<std::vector<int> >& neighbors)
{
    neighbors.resize(keypoints2.size());

    std::vector<Point2f> keypoints1_pts;
    KeyPoint::convert(keypoints1, keypoints1_pts);

    std::vector<Point2f> keypoints2_pts;
    KeyPoint::convert(keypoints2, keypoints2_pts);

    flann::KDTreeIndexParams indexParams;
    Mat input = Mat(keypoints1_pts).reshape(1);
    flann::Index kdtree(input, indexParams);

    std::vector<float> query(2);
    std::vector<int> indices;
    std::vector<float> dists;
    for (size_t i = 0; i < keypoints2_pts.size(); i++)
    {
        query[0] = keypoints2_pts[i].x;
        query[1] = keypoints2_pts[i].y;

        kdtree.knnSearch(query, indices, dists, K);
        neighbors[i] = indices;
    }
}

static void GraCostMatch(int* neighborX, int* neighborY, float lambda, int numNeigh,
                         int numNeighCands, int* Prob, int* p)
{
    if (numNeighCands > numNeigh+1)
    {
        for (int i = 0; i < numNeighCands; i++)
        {
            int cost = 0;
            int ad = 0;
            if (p[i] > 0)
            {
                ad = 1;
            }

            for (int m = 0; m < numNeigh; m++)
            {
                int  num = numNeigh+1;
                int tmp = neighborX[num*i+m+ad];
                bool isMember = false;

                for (int n = 0; n < numNeigh; n++)
                {
                    if (neighborY[num*i+n+ad] == tmp)
                    {
                        isMember = true;
                        break;
                    }
                }
                if (!isMember)
                {
                    cost++;
                }
            }
            cost *= 2;

            if (cost <= lambda)
            {
                Prob[i] = 1;
            }
            else
            {
                Prob[i] = 0;
            }
        }
    }
}

static void GraCostMatch(const std::vector<std::vector<int> >& neighborX, const std::vector<std::vector<int> >& neighborY,
                         float lambda, int numNeigh, int numPoint, int* Prob, int* p)
{
    std::vector<int> new_X(numPoint * (numNeigh+1));
    std::vector<int> new_Y(numPoint * (numNeigh+1));

    for (size_t i = 0; i < neighborX.size(); i++)
    {
        for (size_t j = 0; j < neighborX[i].size(); j++)
        {
            new_X[i*neighborX[i].size() + j] = neighborX[i][j];
        }
    }

    for (size_t i = 0; i < neighborY.size(); i++)
    {
        for (size_t j = 0; j < neighborY[i].size(); j++)
        {
            new_Y[i*neighborY[i].size() + j] = neighborY[i][j];
        }
    }

    GraCostMatch(new_X.data(), new_Y.data(), lambda, numNeigh, numPoint, Prob, p);
}

static bool equal(float val, float comp)
{
    return std::fabs(val-comp) < std::numeric_limits<float>::epsilon();
}

static void find(const std::vector<KeyPoint>& ikpts, std::vector<KeyPoint>& okpts,
                 const std::vector<int>& Prob)
{
    okpts.clear();
    okpts.reserve(ikpts.size());
    for (size_t i = 0; i < Prob.size(); i++)
    {
        if (Prob[i] == 1)
        {
            okpts.push_back(ikpts[i]);
        }
    }
}

static void find(const std::vector<KeyPoint>& kpts, const KeyPoint& kpt, std::vector<int>& p3)
{
    for (size_t i = 0; i < kpts.size(); i++)
    {
        const KeyPoint& k = kpts[i];
        if (equal(k.pt.x, kpt.pt.x) && equal(k.pt.y, kpt.pt.y))
        {
            p3[i] = 1;
        }
    }
}

void matchGLPM(const std::vector<KeyPoint>& keypoints1, const Mat& descriptors1,
               const std::vector<KeyPoint>& keypoints2, const Mat& descriptors2,
               const Ptr<DescriptorMatcher>& matcher,
               std::vector<DMatch>& matches1to2GLPM,
               float LoweRatio)
{
    CV_CheckGT(LoweRatio, 0.0f, "LoweRatio must be > 0");
    CV_CheckLE(LoweRatio, 1.0f, "LoweRatio must be <= 1");
    matches1to2GLPM.clear();

    std::vector<std::vector<DMatch> > knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    std::vector<DMatch> ratioMatches;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i][0].distance / knnMatches[i][1].distance < LoweRatio)
        {
            ratioMatches.push_back(knnMatches[i][0]);
        }
    }

    std::vector<KeyPoint> keypoints1_putative, keypoints2_putative;
    {
        for (size_t i = 0; i < ratioMatches.size(); i++)
        {
            const DMatch& m = ratioMatches[i];
            keypoints1_putative.push_back(keypoints1[m.queryIdx]);
            keypoints2_putative.push_back(keypoints2[m.trainIdx]);
        }
    }

    int numNeighbors1 = 4, numNeighbors2 = 4;
    if (static_cast<int>(keypoints1_putative.size()) < numNeighbors1+1 ||
        static_cast<int>(keypoints2_putative.size()) < numNeighbors1+1)
    {
        return;
    }

    std::vector<std::vector<int> > nn1, nn2;
    nearestNeighbor(keypoints1_putative, keypoints1_putative, numNeighbors1+1, nn1);
    nearestNeighbor(keypoints2_putative, keypoints2_putative, numNeighbors1+1, nn2);

    std::vector<int> p1(nn1.size());
    std::fill(p1.begin(), p1.end(), 1);

    int lambda1 = 6;
    int lambda2 = 4;
    std::vector<int> Prob(p1.size());

    GraCostMatch(nn1, nn2, lambda1, numNeighbors1, static_cast<int>(nn1.size()), Prob.data(), p1.data());

    std::vector<KeyPoint> keypoints1_putative_, keypoints2_putative_;
    find(keypoints1_putative, keypoints1_putative_, Prob);
    find(keypoints2_putative, keypoints2_putative_, Prob);

    if (static_cast<int>(keypoints1_putative_.size()) < numNeighbors2+1 ||
        static_cast<int>(keypoints2_putative_.size()) < numNeighbors2+1)
    {
        return;
    }

    std::vector<std::vector<int> > nn1_, nn2_;
    nearestNeighbor(keypoints1_putative_, keypoints1_putative, numNeighbors2+1, nn1_);
    nearestNeighbor(keypoints2_putative_, keypoints2_putative, numNeighbors2+1, nn2_);

    std::vector<int> p2 = Prob;
    GraCostMatch(nn1_, nn2_, lambda2, numNeighbors2, static_cast<int>(nn1_.size()), Prob.data(), p2.data());

    std::vector<DMatch> nnMatches;
    matcher->match(descriptors1, descriptors2, nnMatches);

    std::vector<KeyPoint> keypoints1_putative2, keypoints2_putative2;
    std::map<int, int> mapKpts1, mapKpts2;
    std::map<int, float> mapDistance;
    for (size_t i = 0; i < nnMatches.size(); i++)
    {
        const DMatch& m = nnMatches[i];
        keypoints1_putative2.push_back(keypoints1[m.queryIdx]);
        mapKpts1[static_cast<int>(i)] = m.queryIdx;

        keypoints2_putative2.push_back(keypoints2[m.trainIdx]);
        mapKpts2[static_cast<int>(i)] = m.trainIdx;

        mapDistance[static_cast<int>(i)] = m.distance;
    }

    std::vector<int> p3(keypoints1_putative2.size());
    std::fill(p3.begin(), p3.end(), 0);
    for (size_t i = 0; i < keypoints1_putative.size(); i++)
    {
        if (p2[i] == 1)
        {
            const KeyPoint& kpt = keypoints1_putative[i];
            find(keypoints1_putative2, kpt, p3);
        }
    }

    std::vector<size_t> indices;
    for (size_t i = 0; i < p3.size(); i++)
    {
        if (p3[i] == 1)
        {
            indices.push_back(i);
        }
    }

    std::vector<KeyPoint> keypoints1_putative2_p3, keypoints2_putative2_p3;
    for (size_t i = 0; i < indices.size(); i++)
    {
        keypoints1_putative2_p3.push_back(keypoints1_putative2[indices[i]]);
        keypoints2_putative2_p3.push_back(keypoints2_putative2[indices[i]]);
    }

    lambda1 = 6;
    numNeighbors1 = 6;

    if (static_cast<int>(keypoints1_putative2_p3.size()) < numNeighbors1+1 ||
        static_cast<int>(keypoints2_putative2_p3.size()) < numNeighbors1+1)
    {
        return;
    }

    std::vector<std::vector<int> > nn12, nn22;
    nearestNeighbor(keypoints1_putative2_p3, keypoints1_putative2, numNeighbors1+1, nn12);
    nearestNeighbor(keypoints2_putative2_p3, keypoints2_putative2, numNeighbors1+1, nn22);

    std::vector<int> Prob2(p3.size());
    GraCostMatch(nn12, nn22, lambda1, numNeighbors1, static_cast<int>(nn12.size()), Prob2.data(), p3.data());

    std::vector<KeyPoint> keypoints1_putative2_Prob2, keypoints2_putative2_Prob2;
    for (size_t i = 0; i < Prob2.size(); i++)
    {
        if (Prob2[i] == 1)
        {
            keypoints1_putative2_Prob2.push_back(keypoints1_putative2[i]);
            keypoints2_putative2_Prob2.push_back(keypoints2_putative2[i]);
        }
    }

    lambda1 = 4;
    numNeighbors1 = 5;

    if (static_cast<int>(keypoints1_putative2.size()) < numNeighbors1+1 ||
        static_cast<int>(keypoints2_putative2.size()) < numNeighbors1+1)
    {
        return;
    }

    std::vector<std::vector<int> > nn12_, nn22_;
    nearestNeighbor(keypoints1_putative2_Prob2, keypoints1_putative2, numNeighbors1+1, nn12_);
    nearestNeighbor(keypoints2_putative2_Prob2, keypoints2_putative2, numNeighbors1+1, nn22_);

    std::vector<int> Prob3(Prob2.size());
    GraCostMatch(nn12_, nn22_, lambda1, numNeighbors1, static_cast<int>(nn12.size()), Prob3.data(), Prob2.data());

    std::vector<size_t> indices_Prob3;
    for (size_t i = 0; i < Prob3.size(); i++)
    {
        if (Prob3[i] == 1)
        {
            indices_Prob3.push_back(i);
            matches1to2GLPM.push_back(DMatch(mapKpts1[static_cast<int>(i)], mapKpts2[static_cast<int>(i)],
                                             mapDistance[static_cast<int>(i)]));
        }
    }
}

} //namespace xfeatures2d
} //namespace cv
