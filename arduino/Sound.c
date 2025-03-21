#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

/*
COMPILE CMD: 
gcc Sound.c -o Sound

VisionInference.py will automatically compile and run the C file, so only need to run VisionInference to test
*/

int main(int argc, char *argv[]) { //Accept command line arguments to find the path
    char exec_path[256];
    char *last_slash;

    // Get absolute path of the running executable
    if (realpath(argv[0], exec_path) == NULL) {
        perror("Error getting executable path to sound");
        return 1;
    }

    // Remove the executable name to get the directory
    last_slash = strrchr(exec_path, '/'); //Find the last '/' in the path to find the filename
    if (last_slash != NULL) {
        *last_slash = '\0'; //Truncate on the working directory
    }

    // Construct the absolute path to the sound file using the extracted path
    char sound_path[512];
    snprintf(sound_path, sizeof(sound_path), "%s/sounds/car_horn_1.mp3", exec_path);

    // Construct the afplay command, run it
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "afplay \"%s\"", sound_path);
    system(cmd);

    return 0;
}