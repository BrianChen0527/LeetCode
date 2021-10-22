using namespace std;
#include <iostream>
#include <vector>
#include <string>
#include <map>


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

//Given a roman numeral, convert it to an integer.
int romanToInt(string s) {
    int sum = 0;
    std::map<char, int> romanNumerals = {
        {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000},
    };
    for (int i = s.length() - 1; i >= 0; i--) {
        int item = romanNumerals.find(s[i])->second;
        int next = 10000;
        if (i != 0)
            next = romanNumerals.find(s[i - 1])->second;
        if (next < item) {
            sum += (item - next);
            i--;
        }
        else
            sum += item;
    }
    return sum;
}

int lengthOfLongestSubstring(string s) {

    int n = s.length();
    int max_len = 0;
    map<char, int> temp;

    for (int i = 0, j = 0; j < n; j++) {
        if (temp.find(s[j]) != temp.end()) {
            i = max(i, temp[s[j]] + 1);
            temp[s[j]] = j;
        }
        max_len = max(max_len, j - i + 1);
        temp.insert({ s[j], j });
    }
    return max_len;
}

//Runtime complexity: O((m+n)/2)
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int merged_length = nums1.size() + nums2.size();
    double median = 0;
    int prev_num = 0;

    for (int i = 0; i < merged_length / 2; i++) {
        if (nums1.size() == 0) {
            prev_num = nums2.at(0);
            nums2.erase(nums2.begin());
        }
        else if (nums2.size() == 0) {
            prev_num = nums1.at(0);
            nums1.erase(nums1.begin());
        }
        else if (nums1.at(0) > nums2.at(0)) {
            prev_num = nums2.at(0);
            nums2.erase(nums2.begin());
        }
        else {
            prev_num = nums1.at(0);
            nums1.erase(nums1.begin());
        }
    }

    if (nums1.size() == 0 && nums2.size() == 0)
        median = 0;
    else if (nums1.size() == 0)
        median = nums2.at(0);
    else if (nums2.size() == 0)
        median = nums1.at(0);
    else
        median = min(nums1.at(0), nums2.at(0));

    if (merged_length % 2 == 0)
        median = (median + prev_num) / 2;

    return median;
}

//Runtime Complexity: O(min(log m, log n))
double findMedianSortedArrays2(vector<int>& nums1, vector<int>& nums2) {

    int n1 = nums1.size();
    int n2 = nums2.size();
    int real_median_pos = (n1 + n2 + 1) / 2;
    if (n1 > n2)
        return findMedianSortedArrays(nums2, nums1);

    int start = 0;
    int end = n1;
    while (start <= end) {
        int mid = (start + end) / 2;
        int n1_median_pos = mid;
        int n2_median_pos = real_median_pos - n1_median_pos;

        int left1 = (n1_median_pos > 0) ? nums1.at(n1_median_pos - 1) : INT_MIN;
        int left2 = (n2_median_pos > 0) ? nums2.at(n2_median_pos - 1) : INT_MIN;
        int right1 = (n1 > n1_median_pos) ? nums1.at(n1_median_pos) : INT_MAX;
        int right2 = (n2 > n2_median_pos) ? nums2.at(n2_median_pos) : INT_MAX;
        if (right2 >= left1 && right1 >= left2) {
            if ((n1 + n2) % 2 == 0) {
                return (max(left1, left2) + min(right1, right2)) / 2.0;
            }
            return max(left1, left2) * 1.0;
        }
        else if (right2 > left1)
            start = mid + 1;
        else
            end = mid - 1;
    }
    return 0.0;
}




















