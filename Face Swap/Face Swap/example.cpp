#include "face_swap.h"

using namespace std;


void example(void)
{
	cv::VideoCapture cap(1);

	cv::Mat cur_frame;
	cap >> cur_frame;

	int cnt = 0;
	while (1)
	{
		cap >> cur_frame;
		cv::flip(cur_frame, cur_frame, 1);
		cv::imshow("output", cur_frame);

		if (cv::waitKey(1) == 27)
			break;
		cout << cnt << endl;
		cnt = cnt + 1;
	}
}
