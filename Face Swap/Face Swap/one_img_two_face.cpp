#include "face_swap.h"

using namespace dlib;
using namespace std;

void one_img_two_face(void)
{
    // Read input images
    string filename = "img/fight_club.jpg";
    cv::Mat img = cv::imread(filename);
 
    //cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0, 0, cv::INTER_LINEAR);
    cv::Mat imgWarped = img.clone();

    array2d<unsigned char> imgDlib;
    assign_image(imgDlib, cv_image<bgr_pixel>(img));

    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    std::vector<cv::Point2f> all_points;

    int flag = 0;
    // Getting points
    flag = two_faceLandmarkDetection(imgDlib, sp, all_points);

    // Slicing points
    std::vector<cv::Point2f> points1 = std::vector<cv::Point2f>(all_points.begin(), all_points.begin() + 67);
    std::vector<cv::Point2f> points2 = std::vector<cv::Point2f>(all_points.begin() + 68, all_points.end());
    
    // Convert Mat to float data type
    img.convertTo(img, CV_32F);
    imgWarped.convertTo(imgWarped, CV_32F);
    
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
    std::vector< std::vector<int> > dt1;
    std::vector< std::vector<int> > dt2;

    cv::Rect rect(0, 0, imgWarped.cols, imgWarped.rows);

    // Detected face points can be out of image range
    try{
        calculateDelaunayTriangles(rect, hull1, dt1);
        calculateDelaunayTriangles(rect, hull2, dt2);
        
        // Apply affine transformation to Delaunay triangles
        for (size_t i = 0; i < dt1.size(); i++)
        {
            std::vector<cv::Point2f> t1, t2;
            // Get points for img1, img2 corresponding to the triangles
            for (size_t j = 0; j < 3; j++)
            {
                t1.push_back(hull1[dt1[i][j]]);
                t2.push_back(hull2[dt1[i][j]]);
            }
        
            warpTriangle(img, imgWarped, t2, t1);
        
        }

        ///////////////////////////////////////////////////////////////////////////////

        for (size_t i = 0; i < dt2.size(); i++)
        {
            std::vector<cv::Point2f> t1, t2;
            // Get points for img1, img2 corresponding to the triangles
            for (size_t j = 0; j < 3; j++)
            {
                t1.push_back(hull1[dt2[i][j]]);
                t2.push_back(hull2[dt2[i][j]]);
            }

            warpTriangle(img, imgWarped, t1, t2);
        }


        // Calculate mask
        std::vector<cv::Point> hull8U_1;
        std::vector<cv::Point> hull8U_2;

        for (int i = 0; i < hull2.size(); i++)
        {
            cv::Point pt1(hull1[i].x, hull1[i].y);
            cv::Point pt2(hull2[i].x, hull2[i].y);
            hull8U_1.push_back(pt1);
            hull8U_2.push_back(pt2);
        }
        
        imgWarped.convertTo(imgWarped, CV_8UC3);
        img.convertTo(img, CV_8UC3);

        cv::Mat mask1 = cv::Mat::zeros(img.rows, img.cols, img.depth());
        cv::Mat mask2 = cv::Mat::zeros(img.rows, img.cols, img.depth());

        fillConvexPoly(mask1, &hull8U_1[0], hull8U_1.size(), cv::Scalar(255, 255, 255));
        fillConvexPoly(mask2, &hull8U_2[0], hull8U_2.size(), cv::Scalar(255, 255, 255));

        // Clone seamlessly.
        cv::Rect r1 = boundingRect(hull1);
        cv::Rect r2 = boundingRect(hull2);

        cv::Point center1 = (r1.tl() + r1.br()) / 2;
        cv::Point center2 = (r2.tl() + r2.br()) / 2;

        cv::Mat tmp_output;
        cv::Mat output;

        cv::seamlessClone(imgWarped, img, mask1, center1, tmp_output, cv::NORMAL_CLONE);
        cv::seamlessClone(imgWarped, tmp_output, mask2, center2, output, cv::NORMAL_CLONE);
        
        cv::imshow("Face Swapped", output);
        cv::waitKey(0);
    }
    catch (exception e)
    {
        cout << "Points out of range or Not enough face" << endl;
        cv::imshow("output", img);
    }

    cv::destroyAllWindows();
    
    return;
}