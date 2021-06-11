#include "face_swap.h"


using namespace dlib;
using namespace std;


void realtime_twoface(void)
{
	cv::VideoCapture cap(1);

	if (!cap.isOpened())
	{
		cout << "Can't open the camera" << endl;
		return;
	}
	array2d<unsigned char> cur_Dlib;
	shape_predictor sp;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	cv::Mat imgWarped;
	cv::Mat cur_frame;

	cap >> cur_frame;

	while (1)
	{
		std::vector<cv::Point2f> all_points, hull1, hull2;
		std::vector<int> hullIndex;
		std::vector< std::vector<int> > dt1, dt2;
		std::vector<cv::Point> hull8U_1, hull8U_2;
		cv::Rect r1, r2;
		cv::Point center1, center2;
		cv::Mat mask1, mask2;
		cv::Mat tmp_output, output;

		cap >> cur_frame;
		cv::flip(cur_frame, cur_frame, 1);
		//cv::imshow("Camera img", cur_frame);
		assign_image(cur_Dlib, cv_image<bgr_pixel>(cur_frame));

		int flag = 1;
		// if flag is 1, then face is detected. 0 on the other hand mean vise versa.
		flag = two_faceLandmarkDetection(cur_Dlib, sp, all_points);

		if (flag == 1)
		{
			imgWarped = cur_frame.clone();

			// Slicing points
			std::vector<cv::Point2f> points1 = std::vector<cv::Point2f>(all_points.begin(), all_points.begin() + 67);
			std::vector<cv::Point2f> points2 = std::vector<cv::Point2f>(all_points.begin() + 68, all_points.end());

			// Convert Mat to float data type
			cur_frame.convertTo(cur_frame, CV_32F);
			imgWarped.convertTo(imgWarped, CV_32F);

			// Find convex hull
			cv::convexHull(points2, hullIndex, false, false);

			for (int i = 0; i < hullIndex.size(); i++)
			{
				hull1.push_back(points1[hullIndex[i]]);
				hull2.push_back(points2[hullIndex[i]]);
			}

			// Find delaunay triangulation for points on the convex hull
			cv::Rect rect(0, 0, imgWarped.cols, imgWarped.rows);

			// Detected face points can be out of image range.
			try {
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

					warpTriangle(cur_frame, imgWarped, t2, t1);

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

					warpTriangle(cur_frame, imgWarped, t1, t2);
				}

				// Calculate mask
				for (int i = 0; i < hull2.size(); i++)
				{
					cv::Point pt1(hull1[i].x, hull1[i].y);
					cv::Point pt2(hull2[i].x, hull2[i].y);
					hull8U_1.push_back(pt1);
					hull8U_2.push_back(pt2);
				}

				imgWarped.convertTo(imgWarped, CV_8UC3);
				cur_frame.convertTo(cur_frame, CV_8UC3);

				mask1 = cv::Mat::zeros(cur_frame.rows, cur_frame.cols, cur_frame.depth());
				mask2 = cv::Mat::zeros(cur_frame.rows, cur_frame.cols, cur_frame.depth());

				fillConvexPoly(mask1, &hull8U_1[0], hull8U_1.size(), cv::Scalar(255, 255, 255));
				fillConvexPoly(mask2, &hull8U_2[0], hull8U_2.size(), cv::Scalar(255, 255, 255));

				// Clone seamlessly.
				cv::Rect r1 = boundingRect(hull1);
				cv::Rect r2 = boundingRect(hull2);

				cv::Point center1 = (r1.tl() + r1.br()) / 2;
				cv::Point center2 = (r2.tl() + r2.br()) / 2;

				cv::seamlessClone(imgWarped, cur_frame, mask1, center1, tmp_output, cv::NORMAL_CLONE);
				cv::seamlessClone(imgWarped, tmp_output, mask2, center2, output, cv::NORMAL_CLONE);
				
				cv::imshow("output", output);
				
			}
			catch (exception e)
			{
				cout << "Points out of range or Not enough face" << endl;
				cap >> cur_frame;
				cv::flip(cur_frame, cur_frame, 1);
				cv::imshow("output", cur_frame);

			}

		}
		else // When flag is 0 ( No face detected )
		{
			cv::imshow("output", cur_frame);
		}
		if (cv::waitKey(1) == 27)
			break;
	}
	return;
}