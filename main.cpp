using namespace std;
#include <functional>
#include <queue>
#include <stack>
#include <vector>
#include <iostream>
#include "Functions.h"
#include <fstream>

void printTree(TreeNode* root) {
	queue<TreeNode*> q;
	q.push(root);

	while (q.size()) {
		queue<TreeNode*> q2;

		while (q.size()) {
			TreeNode* node = q.front();
			cout << node->val << ' ';
			if (node->left) q2.push(node->left);
			if (node->right) q2.push(node->right);
			q.pop();
		}
		cout << endl;
		q = q2;
	}
}

template<typename T>
void vector_print(vector<T> v) {
	for (auto i : v) {
		cout << i << ' ';
	}
	cout << endl;
}

class CoordCompare
{
public:
	bool operator() (vector<int> p1, vector<int> p2)
	{
		return (pow(p1[0], 2) + pow(p1[1], 2)) < (pow(p2[0], 2) + pow(p2[1], 2));
	}
};


void inOrderTraversalPrint(TreeNode* root) {
	if (root->left) inOrderTraversalPrint(root->left);
	cout << root->val << ' ';
	if (root->right) inOrderTraversalPrint(root->right);
}
#include <typeinfo>  //for 'typeid' to work  

class MedianFinder {
public:
	priority_queue<int> minQ;
	priority_queue<int> maxQ;

	MedianFinder() {}

	void addNum(int num) {

		priority_queue<int> minQ1 = minQ;
		priority_queue<int> maxQ1 = maxQ;
		while (!minQ1.empty()) {
			cout << minQ1.top() << endl;
			minQ1.pop();
		}
		cout << "===========\n";
		while (!maxQ1.empty()) {
			cout << maxQ1.top() << endl;
			maxQ1.pop();
		}
		cout << "===============\n";


		if (minQ.size() > maxQ.size()) {
			if (num > -minQ.top()) {
				maxQ.push(-minQ.top());
				minQ.pop();
				minQ.push(-num);
			}
			else maxQ.push(num);
		}
		else {
			if (minQ.empty() || num > -minQ.top()) minQ.push(-num);
			else {
				maxQ.push(num);
				minQ.push(-maxQ.top());
				maxQ.pop();
			}
		}
	}

	double findMedian() {

		priority_queue<int> minQ1 = minQ;
		priority_queue<int> maxQ1 = maxQ;
		while (!minQ1.empty()) {
			cout << minQ1.top() << endl;
			minQ1.pop();
		}
		cout << "===========\n";
		while (!maxQ1.empty()) {
			cout << maxQ1.top() << endl;
			maxQ1.pop();
		}
		cout << "===============\n";
		
		if (minQ.size() > maxQ.size()) { return -minQ.top(); }
		else if (maxQ.size() > minQ.size()) { return maxQ.top(); }
		else { 
			cout << maxQ.top()-minQ.top() << " jerere\n";
			return maxQ.top() - minQ.top() / 2.0; }
	}
};


int main() { 
	vector<string> strList = { "hot","dot","dog","lot","log","cog" };
	cout << ladderLength("hit", "cog", strList);

}



