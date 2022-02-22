using namespace std;

#include <map>
#include <algorithm>
#include <cassert>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// Given two strings s and t, return true if t is an anagram of s, and false otherwise.
bool isAnagram(string s, string t) {
    if (s.length() != t.length())
        return false;
    int letters[26] = { 0 };
    for (char i : s) {
        letters[i - 'a']++;
    }
    for (char i : t) {
        letters[i - 'a']--;
    }
    for (int i = 0; i < 26; i++) {
        if (letters[i] != 0) {
            return false;
        }
    }
    return true;
}

// EECS 281 lab 4
// Given n ropes, find the minimum cost of connecting ropes where the cost of connecting two
// ropes is the sum of their lengths.
// Example : If you had 10, 5, 8, 14, the min cost is 73.
// Explanation : Join ropes 5 and 8 to get a rope of length 13 (Net cost = 13) Join ropes 13
// and 10 to get a rope of length 23 (Net cost = 13 + 23) Join ropes 23 and 14 to get a rope
// of length 37 (Net cost = 13 + 23 + 37 = 73)
// https://umich.instructure.com/courses/491173/files/folder/Lab/lab04/Lab%20Assignment%20(Quiz)?preview=24159867
int join_ropes(const vector<int>& rope_lengths) {
    
    int min_length = 0;
    vector<int> total;
    if (rope_lengths.size() == 1) {
        return rope_lengths.at(0);
    }

    for (int i = 0; i < rope_lengths.size(); i++) {
        total.push_back(rope_lengths.at(i));
    }
    sort(total.begin(), total.end());

    while (total.size() > 1) {
        int new_num = total.at(0) + total.at(1);
        min_length += new_num;

        if (total.size() == 2) {
            break;
        }

        for (int i = 2; i < total.size(); i++) {
            if (new_num < total.at(i)) {
                total.insert(total.begin() + i, new_num);
                break;
            }
            if (i == total.size() - 1) {
                total.push_back(new_num);
                break;
            }
        }

        total.erase(total.begin());
        total.erase(total.begin());

    }


    return min_length;
}

//################################################################################
//################################################################################
//###########################                       ##############################
//###########################   EECS 281 Practice   ##############################
//###########################                       ##############################
//################################################################################
//################################################################################

struct Node {
    int val;
    Node* next;
    Node() : val{ 0 }, next{ nullptr } {}
    Node(int x) : val{ x }, next{ nullptr } {}
    Node(int x, Node* next_in) : val{ x }, next{ next_in } {}
};


// Write a program that reverses this singly-linked list of nodes
Node* reverse_list(Node* head) {
    Node* prev = nullptr;

    if (!head) {
        return nullptr;
    }
    while (head) {

        Node* next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}


// Write a function that deletes all duplicates in the list so that 
// each element ends up only appearing once.
Node* remove_duplicates(Node* head) {
    if (!head) {
        return nullptr;
    }
    Node* current = head;

    while (current && current->next) {
        if (current->val == current->next->val) {
            Node* temp = current->next;
            current->next = current->next->next;
            delete temp;
        }
        else {
            current = current->next;
        }
    }
    return head;
}


// You are given a non-empty vector of distinct elements, and you want to
// return a vector that stores the previous greater element that exists
// before each index.If no previous greater element exists, -1 is stored.
vector<int> prev_greatest_element(vector<int>& vec) {
    int prev_greater = -1;
    if (vec.size() > 1) {
        vector<int> tmp;

        for (int i = 0; i < vec.size() - 1; i++) {
            tmp.push_back(prev_greater);
            if (vec.at(i + 1) < vec.at(i)) {
                prev_greater = vec.at(i);
            }
        }
        tmp.push_back(prev_greater);
        return tmp;
    }
    else if (vec.size() == 1) {
        return vector<int> {-1};
    }
    else {
        return vector<int> {};
    }

}


// You are given a collection of intervals. Write a function that 
// merges all of the overlapping intervals.
struct Interval {
    int start;
    int end;
};
bool cmp_Intervals(const Interval& a, const Interval& b) { return (a.start < b.start); }
vector<Interval> merge_intervals1(vector<Interval>& vec) {

    

    sort(vec.begin(), vec.end(), cmp_Intervals);

    vector<Interval> sorted;
    if (vec.size() > 1) {
        sorted.push_back(vec.at(0));

        for (int i = 1; i < vec.size(); i++) {
            if (sorted.at(sorted.size() - 1).end < vec.at(i).start) {
                sorted.push_back(vec.at(i));
            }
            else {
                int end = sorted.at(sorted.size() - 1).end;
                sorted.at(sorted.size() - 1).end = max(end, vec.at(i).end);
            }
        }
    }
    else {
        sorted = vec;
    }
    return sorted;
}


// You are given two non-empty linked lists representing two non-negative integers. The most significant
// digit comes firstand each of their nodes contains a single digit. Add the two numbersand return the result
// as a linked list.You may assume the two numbers do not contain any leading 0’s except the number 0
// itself.The structure of a Node is provided below :

Node* add_lists(Node* list1, Node* list2) {
    deque<int> num1, num2;
    int ans = 0;
    Node* tmp1 = list1;
    Node* tmp2 = list2;

    while (tmp1) {
        num1.push_front(tmp1->val);
        tmp1 = tmp1->next;
    }
    while (tmp2) {
        num2.push_front(tmp2->val);
        tmp2 = tmp2->next;
    }

    int max_size = max(num1.size(), num2.size());

    for (int i = 0; i < max_size; i++) {
        int num = 0;
        if (num1.size() > i && num2.size() > i) {
            num = num1.at(i) + num2.at(i);
        }
        else if (num1.size() > i) { num = num1.at(i); }
        else { num = num2.at(i); }

        ans += (num * pow(10, i));
    }

    Node* head = new Node(ans / pow(10, max_size - 1));
    Node* curr = head;

    for (int i = max_size - 2; i >= 0; i--) {
        ans %= int(pow(10, i + 1));
        curr->next = new Node(int(ans / pow(10, i)));
        curr = curr->next;
    }
    return head;
}


// Suppose you are given an array of int, size n and a number k. Return the k largest elements.
// Output does not need to be sorted.You can assume that k < n.
vector<int> findKMax(int arr[], size_t n, size_t k) {

    priority_queue<int, vector<int>, greater<int>> top_k;
    for (int i = 0; i < k; i++) {
        top_k.push(arr[i]);
    }

    for (int i = k; i < n; i++) {
        if (arr[i] > top_k.top()) {
            top_k.pop();
            top_k.push(arr[i]);
        }
    }
    cout << top_k.size() << endl;

    vector <int> ans;
    while (!top_k.empty()) {
        ans.push_back(top_k.top());
        top_k.pop();
    }
    return ans;

}


// Time Complexity Restriction: O(logn)
// Space Complexity Restriction: O(1)
// You are given a sorted array consisting of only integers where every element appears exactly twice, except
// for one element which appears exactly once.Write a function that returns the single element.
int find_single_element(vector<int>& vec) {
    int left = 0;
    int right = vec.size() - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (vec.at(mid) == vec.at(mid + 1)) {
            (mid % 2 == 0) ? (left = mid) : (right = mid - 1);
        }
        else if (vec.at(mid) == vec.at(mid - 1)) {
            (mid % 2 == 0) ? (right = mid) : (left = mid + 1);
        }
    }
    return vec.at(left);
}


Node* merge_lists(vector<Node*>& lists) {
    while (lists.size() > 1) {
        // merge 2 lists together
        vector<Node*> merged_list;
        Node* list1 = lists.at(0);
        Node* list2 = lists.at(1);

        while (list1 || list2) {
            if (!list2) {
                merged_list.push_back(list1);
                list1 = list1->next;
            }
            else if (!list1) {
                merged_list.push_back(list2);
                list2 = list2->next;
            }
            else {
                if (list1->val > list2->val) {
                    merged_list.push_back(list2);
                    list2 = list2->next;
                }
                else {
                    merged_list.push_back(list1);
                    list1 = list1->next;
                }
            }
        }
        for (int i = 0; i < merged_list.size() - 1; i++) {
            merged_list.at(i)->next = merged_list.at(i + 1);
        }
        // pop the merged lists from the vector
        lists.erase(lists.begin(), lists.begin() + 2);

        // push merged list to the back of lists
        lists.push_back(merged_list.at(0));
    }
    return lists.at(0);
}
