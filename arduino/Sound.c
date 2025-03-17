#include <stdlib.h>

/*
COMPILE CMD: 
gcc Sound.c -o Sound

VisionInference.py will automatically compile and run the C file, so only need to run VisionInference to test
*/

int main() {
    system("afplay ../model/sounds/car_horn_1.mp3");
    return 0;
}