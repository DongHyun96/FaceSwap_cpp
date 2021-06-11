#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>

#include <fstream>
#include <string>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <dlib/opencv/cv_image.h>


// Main Running Functions //

void face_swap_origin(void);
void cam_detection(void);
void img_detection(void);
void arbitrary_source_swap(void);
void one_img_two_face(void);
void realtime_swap(void);
void realtime_twoface(void);

void example(void);



// Additional Functions //

// From core_functions.cpp
std::vector<cv::Point2f> readPoints(std::string pointsFileName);
void applyAffineTransform(cv::Mat& warpImage, cv::Mat& src, std::vector<cv::Point2f>& srcTri, std::vector<cv::Point2f>& dstTri);
void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector< std::vector<int> >& delaunayTri);
void warpTriangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& t1, std::vector<cv::Point2f>& t2);

// From image_facemark.cpp
int single_faceLandmarkDetection(dlib::array2d<unsigned char>& img, dlib::shape_predictor sp, std::vector<cv::Point2f>& landmark);
int two_faceLandmarkDetection(dlib::array2d<unsigned char>& img, dlib::shape_predictor sp, std::vector<cv::Point2f>& landmark);
