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
			cout << node->val << " ";
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
		cout << i << " ";
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
	cout << root->val << " ";
	if (root->right) inOrderTraversalPrint(root->right);
}

int main() { 
	TreeNode* t1 = new TreeNode(1);
	TreeNode* t2 = new TreeNode(2);
	TreeNode* t3 = new TreeNode(3);

	vector<int> preorder = { 3,9,20,15,7 };
	vector<int> inorder = { 9,3,15,20,7 };

	TreeNode* t = buildTree(preorder, inorder);

	inOrderTraversalPrint(t);
}
