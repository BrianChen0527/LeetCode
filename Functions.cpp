using namespace std;
#include <iostream>
#include <vector>
#include <string>

//Given an array of integers numsand an integer target, return indices of the two numbers such that they add up to target
vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> ans = {};
    for (int i = 0; i < nums.size(); i++) {
        for (int j = i; j < nums.size(); j++) {
            if (nums.at(i) + nums.at(j) == target && i != j) {
                ans.push_back(i);
                ans.push_back(j);
            }
        }
    }
    return ans;
}


//Given an integer x, return true if x is palindrome integer.
bool isPalindrome(int x) {
    string str_x = to_string(x);
    for (int i = 0; i < str_x.length() / 2; i++) {
        if (str_x[i] != str_x[str_x.length() - i - 1]) {
            return false;
        }
    }
    return true;
}




























