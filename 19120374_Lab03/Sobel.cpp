#include "Sobel.h"

vector<vector<char>> sobel_filter3x3_x = { {-1,0,1},{-2,0,2},{-1,0,1} };
vector<vector<char>> sobel_filter3x3_y = { {1,2,1},{0,0,0},{-1,-2,-1} };
vector<vector<char>> sobel_filter5x5_x = { {2,2,4,2,2}, {1,1,2,1,1},{0,0,0,0,0},{-1,-1,-2,-1,-1},{-2,-2,-4,-2,-2} };
vector<vector<char>> sobel_filter5x5_y = { {2,1,0,-1,-2}, {2,1,0,-1,-2},{4,2,0,-2,-4},{2,1,0,-1,-2},{2,1,0,-1,-2} };