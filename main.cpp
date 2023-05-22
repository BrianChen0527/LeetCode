using namespace std;
#include <functional>
#include <queue>
#include <stack>
#include <vector>
#include <iostream>
#include "Functions.h"
#include <fstream>

void printTree(TreeNode* root) {
	queue< TreeNode*> q;
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

int main() { 

	vector<int>  preorder = { 5, 3, 2, 1, 4, 6 };
	vector<int> inorder = {1,2,3,4,5,6};
	TreeNode* root = buildTree(preorder, inorder);
	printTree(root);
	cout << kthSmallest(root, 3);
}
