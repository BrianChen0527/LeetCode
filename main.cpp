using namespace std;
#include "Functions.h"


int main() {
	vector<int> nums = { 10,20,30,40,50,60,70,80,90,11 };
	int target = 99;
	cout << combinationSum4(nums, target);
}