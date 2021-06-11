#include "face_swap.h"

using namespace std;

/*
 *   You can test each functions one by one
 *   Debug mode 가 아닌 Release mode로 실행해주세요.
 */
int main(int argc, char** argv)
{
    //face_swap_origin(); // Basic swap function.

    //arbitrary_source_swap(); // Swap arbitrary images.

    one_img_two_face(); // Swap two faces in one image.

    //realtime_swap(); // You can test this through web cam. Swap your face with arbitrary source in realTime.

    //realtime_twoface(); // Test this with web cam. Needs two peoeple to swap each other's face.

    return 1;
}
