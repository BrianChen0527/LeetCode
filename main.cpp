using namespace std;
#include <iostream>
#include <vector>
#include <string>
#include "Functions.h"



int main(int argc, char* argv[]){
	vector<int> nums = { 1,2,7,9 };
	vector<int> ans = twoSum(nums, 9);
	for (auto i: ans) {
		cout << i;
	}

}

