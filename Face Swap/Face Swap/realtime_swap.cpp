#include "face_swap.h"


using namespace dlib;
using namespace std;


void realtime_swap(void)
{
	cv::VideoCapture cap(1);

	if (!cap.isOpened())
	{
		cout << "Can't open the camera" << endl;
		return;
	}

	// Get source image
	string filename = "img/iu.jpg";
	cv::Mat src_img = cv::imread(filename);
	
	array2d<unsigned char> srcDlib, cur_Dlib;
	assign_image(srcDlib, cv_image<bgr_pixel>(src_img));

	// Convert Mat to float data type
	src_img.convertTo(src_img, CV_32F);

	shape_predictor sp;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	cv::Mat imgWarped;
	std::vector<cv::Point2f> src_points;

	// Getting points
	single_faceLandmarkDetection(srcDlib, sp, src_points);

	cv::Mat cur_frame;

	cap >> cur_frame;

	while (1)
	{
		std::vector<cv::Point2f> cur_points, hull1, hull2;
		std::vector<int> hullIndex;
		std::vector< std::vector<int> > dt;
		std::vector<cv::Point> hull8U;
		cv::Rect r;
		cv::Point center;
		cv::Mat mask;
		cv::Mat output;

		cap >> cur_frame;
		cv::flip(cur_frame, cur_frame, 1);
		//cv::imshow("Camera img", cur_frame);
		assign_image(cur_Dlib, cv_image<bgr_pixel>(cur_frame));

		int flag = 1;
		// if flag is 1, then face is detected. 0 on the other hand mean vise versa.
		flag = single_faceLandmarkDetection(cur_Dlib, sp, cur_points);
		
		if (flag == 1)
		{
			imgWarped = cur_frame.clone();

			// Convert Mat to float data type
			imgWarped.convertTo(imgWarped, CV_32F);

			// Find convex hull
			cv::convexHull(cur_points, hullIndex, false, false);

			for (int i = 0; i < hullIndex.size(); i++)
			{
				hull1.push_back(src_points[hullIndex[i]]);
				hull2.push_back(cur_points[hullIndex[i]]);
			}

			// Find delaunay triangulation for points on the convex hull
			cv::Rect rect(0, 0, imgWarped.cols, imgWarped.rows);

			// Detected face points can be out of image range.
			try{
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

					warpTriangle(src_img, imgWarped, t1, t2);
				}

				// Calculate mask
				for (int i = 0; i < hull2.size(); i++)
				{
					cv::Point pt(hull2[i].x, hull2[i].y);
					hull8U.push_back(pt);
				}

				mask = cv::Mat::zeros(cur_frame.rows, cur_frame.cols, cur_frame.depth());
				fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

				// Clone seamlessly.
				r = boundingRect(hull2);
				center = (r.tl() + r.br()) / 2;

				imgWarped.convertTo(imgWarped, CV_8UC3);
				seamlessClone(imgWarped, cur_frame, mask, center, output, cv::NORMAL_CLONE);
				cv::imshow("output", output);
			}
			catch (exception e) 
			{
				cout << "Points out of range" << endl;
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
