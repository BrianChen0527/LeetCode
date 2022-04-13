using namespace std;
#include "Functions.h"


int main() {
	vector<int> t = { 3, 2, 5};
	int target = 76;
	for (int i : howSum(target, t)) {
		cout << i << " ";
	}
}