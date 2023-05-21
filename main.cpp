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

class LRUCache {
public:
    unordered_map<int, pair<int, int>> key_value;
    unordered_map<int, int> timestamp_key;
    int timestamp = 0; int items = 0; int oldest = 0; int cap = 0;

    LRUCache(int capacity) {
        cap = capacity;
    }

    int get(int key) {
        if (key_value.find(key) == key_value.end()) return -1;
        int old_timestamp = key_value[key].second;

        key_value[key].second = timestamp;
        timestamp_key.erase(old_timestamp);
        timestamp_key[timestamp] = key;
        timestamp++;

        while (timestamp_key.find(oldest) == timestamp_key.end()) oldest++;
        return key_value[key].first;
    }

    void put(int key, int value) {
        if (key_value.find(key) != key_value.end()) {
            int old_timestamp = key_value[key].second;
            timestamp_key.erase(old_timestamp);
            items--;
        }
        else if (items >= cap) {
            int old_key = timestamp_key[oldest];

            key_value.erase(old_key);
            timestamp_key.erase(oldest);

        }
        key_value[key] = make_pair(value, timestamp);
        timestamp_key[timestamp] = key;
        timestamp++;
        items++;

        while (timestamp_key.find(oldest) == timestamp_key.end()) oldest++;
    }
};

int main() { 
	LRUCache test(2);

	test.put(1, 1);
	test.put(2, 2);
    test.get(1);
	test.put(3, 3);
    cout << test.get(2);
}
