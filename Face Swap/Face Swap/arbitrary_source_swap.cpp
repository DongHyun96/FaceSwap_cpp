#include "face_swap.h"

using namespace dlib;
using namespace std;

void arbitrary_source_swap(void)
{
    // Read input images
    string filename1 = "img/iu.jpg";
    string filename2 = "img/my_pictures.jpg";
    cv::Mat img1 = cv::imread(filename1);
    cv::Mat img2 = cv::imread(filename2);
    cv::resize(img2, img2, cv::Size(img2.cols / 4, img2.rows / 4), 0, 0, cv::INTER_LINEAR);

    cv::Mat img1Warped = img2.clone();

    array2d<unsigned char> imgDlib1, imgDlib2;
    assign_image(imgDlib1, cv_image<bgr_pixel>(img1));
    assign_image(imgDlib2, cv_image<bgr_pixel>(img2));

    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    std::vector<cv::Point2f> points1, points2;

    int flag = 0;
    // Getting points
    flag = single_faceLandmarkDetection(imgDlib1, sp, points1);
    flag = single_faceLandmarkDetection(imgDlib2, sp, points2);

    // Convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);


    // Find convex hull
    std::vector<cv::Point2f> hull1;
    std::vector<cv::Point2f> hull2;
    std::vector<int> hullIndex;

    cv::convexHull(points2, hullIndex, false, false);

    for (int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }


    // Find delaunay triangulation for points on the convex hull
    std::vector< std::vector<int> > dt;
    cv::Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
    calculateDelaunayTriangles(rect, hull2, dt);

    // Apply affine transformation to Delaunay triangles
    for (size_t i = 0; i < dt.size(); i++)
    {
        std::vector<cv::Point2f> t1, t2;
        // Get points for img1, img2 corresponding to the triangles
        for (size_t j = 0; j < 3; j++)
        {
            t1.push_back(hull1[dt[i][j]]);
            t2.push_back(hull2[dt[i][j]]);
        }

        warpTriangle(img1, img1Warped, t1, t2);

    }

    // Calculate mask
    std::vector<cv::Point> hull8U;
    for (int i = 0; i < hull2.size(); i++)
    {
        cv::Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    cv::Mat mask = cv::Mat::zeros(img2.rows, img2.cols, img2.depth());
    fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

    // Clone seamlessly.
    cv::Rect r = boundingRect(hull2);
    cv::Point center = (r.tl() + r.br()) / 2;

    cv::Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
    seamlessClone(img1Warped, img2, mask, center, output, cv::NORMAL_CLONE);

    cv::imshow("Face Swapped", output);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return;
}