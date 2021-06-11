#include "face_swap.h"

using namespace dlib;
using namespace std;


frontal_face_detector detector = get_frontal_face_detector();


/*
//detect 68 face landmarks on the input image by using the face landmark detector in dlib.
*/
int single_faceLandmarkDetection(array2d<unsigned char>& img, shape_predictor sp, std::vector<cv::Point2f>& landmark)
{
    int flag = 1;  // if flag is 1, then face is detected. 0 on the other hand mean vise versa.
    std::vector<rectangle> dets = detector(img);

    if (dets.empty()) {
        cout << "No face is deteced by DLib" << endl;
        flag = 0;
        return flag;
    }

    full_object_detection shape = sp(img, dets[0]);

    for (int i = 0; i < shape.num_parts(); ++i)
    {
        float x = shape.part(i).x();
        float y = shape.part(i).y();
        landmark.push_back(cv::Point2f(x, y));
    }

    return flag;
}

int two_faceLandmarkDetection(array2d<unsigned char>& img, shape_predictor sp, std::vector<cv::Point2f>& landmark)
{
    int flag = 1;  // if flag is 1, then face is detected. 0 on the other hand mean vise versa.
    std::vector<rectangle> dets = detector(img);

    if (dets.empty()) {
        cout << "No face is deteced by DLib" << endl;
        flag = 0;
        return flag;
    }
    full_object_detection shape = sp(img, dets[0]);
    full_object_detection shape2 = sp(img, dets[1]);

    for (int i = 0; i < shape.num_parts(); ++i)
    {
        float x = shape.part(i).x();
        float y = shape.part(i).y();
        landmark.push_back(cv::Point2f(x, y));
    }

    for (int i = 0; i < shape2.num_parts(); ++i)
    {
        float x = shape2.part(i).x();
        float y = shape2.part(i).y();
        landmark.push_back(cv::Point2f(x, y));
    }
    return flag;
}

void img_detection(void)
{
    // Load face detection and pose estimation models.
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    std::vector<cv::Point2f> points;

    cv::Mat img = cv::imread("img/my_pictures.jpg");
    cv::resize(img, img, cv::Size(img.cols / 4, img.rows / 4), 0, 0, cv::INTER_LINEAR);
    cv::Mat temp = img.clone();

    array2d<unsigned char> imgDlib;
    assign_image(imgDlib, cv_image<bgr_pixel>(temp));

    single_faceLandmarkDetection(imgDlib, sp, points);

    for (std::vector<cv::Point2f>::size_type i = 0; i < points.size(); ++i)
    {
        cv::circle(img, points[i], 3, cv::Scalar(255, 0, 0), -1);
    }

    cv::namedWindow("img", cv::WINDOW_KEEPRATIO);
    cv::imshow("img", img);
    cv::waitKey(0);
}