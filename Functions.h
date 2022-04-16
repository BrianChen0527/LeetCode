using namespace std;
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
//Given an array of integers numsand an integer target, return indices of the two numbers such that they add up to target
vector<int> twoSum(vector<int>& nums, int target);

//Given an integer x, return true if x is palindrome integer.
bool isPalindrome(int x);

//Given a roman numeral, convert it to an integer.
int romanToInt(string s);

//Given a string s, find the length of the longest substring without repeating characters.
int lengthOfLongestSubstring(string s);

//Given a string s, find the length of the longest substring without repeating characters.
int lengthOfLongestSubstring(string s);

//Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);

//Improved version of findMedianSortedArrays()
double findMedianSortedArrays2(vector<int>& nums1, vector<int>& nums2);

// Given a string s, return the longest palindromic substring in s.
string longestPalindrome(string s);

// Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
// https://leetcode.com/problems/subarray-sum-equals-k/
int subarraySum(vector<int>& nums, int k);


struct Node {
    int val;
    Node* next;
    Node() : val{ 0 }, next{ nullptr } {}
    Node(int x) : val{ x }, next{ nullptr } {}
    Node(int x, Node* next_in) : val{ x }, next{ next_in } {}
};

// Write a program that reverses this singly-linked list of nodes
Node* reverse_list(Node* head);

// Write a function that deletes all duplicates in the list so that 
// each element ends up only appearing once.
Node* remove_duplicates(Node* head);

// You are given a non-empty vector of distinct elements, and you want to
// return a vector that stores the previous greater element that exists
// before each index.If no previous greater element exists, -1 is stored.
vector<int> prev_greatest_element(vector<int>& vec);

// You are given a collection of intervals. Write a function that 
// merges all of the overlapping intervals.
struct Interval {
    int start;
    int end;
};
bool cmp_Intervals(const Interval& a, const Interval& b);
vector<Interval> merge_intervals1(vector<Interval>& vec);

// You are given two non-empty linked lists representing two non-negative integers. The most significant
// digit comes firstand each of their nodes contains a single digit. Add the two numbersand return the result
// as a linked list.You may assume the two numbers do not contain any leading 0’s except the number 0
// itself.The structure of a Node is provided below :
Node* add_lists(Node* list1, Node* list2);

// Suppose you are given an array of int, size n and a number k. Return the k largest elements.
// Output does not need to be sorted.You can assume that k < n.
vector<int> findKMax(int arr[], size_t n, size_t k);

// Time Complexity Restriction: O(logn)
// Space Complexity Restriction: O(1)
// You are given a sorted array consisting of only integers where every element appears exactly twice, except
// for one element which appears exactly once.Write a function that returns the single element.
int find_single_element(vector<int>& vec);

// You are given k sorted linked lists. Write a program that merges all k lists into a single sorted list.
Node* merge_lists(vector<Node*>& lists);

// You are given a vector of integers, vec, and you are told to implement a function that moves all elements
// with a value of 0 to the end of the vector while maintaining the relative order of the non - zero elements.
void shift_zeros(vector<int>& vec);

// You are given a vector of integers, temps, that stores the daily temperature forecasts for the next few
// days.Write a program that, for each index of the input vector, stores the number of days you need to wait
// for a warmer temperature.If there is no future day where this is possible, a value of 0 should be stored.
vector<int> warmer_temperatures(vector<int>& temps);

// You are given a m x n matrix in the form of a vector of vectors that has the following properties:
// • integers in each row are sorted in ascending order from left to right
// • integers in each column are sorted in ascending order from top to bottom
// Write a function that searches for a value in this matrixand returns whether the element can be found.
bool matrix_search(vector<vector<int>>& matrix, int target);

// EECS 281 Lab7 Written Problem
// https://umich.instructure.com/courses/491173/files/folder/Lab/lab07/Replace%20Words%20Written%20Problem?preview=23559563
vector<string> replace_words(const vector<string>& prefixes, const vector<string>& sentence);

// find the longest palindromic substring in a string
string longestPalindrome(string s);

// Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
// https://leetcode.com/problems/subarray-sum-equals-k/
int subarraySum(vector<int>& nums, int k);

// Given an unsorted array of integers nums, 
// return the length of the longest consecutive elements sequence.
int longestConsecutive(vector<int>& nums);


int longestConsecutive(vector<int>& nums);

int reverse(int x);

// https://leetcode.com/problems/string-to-integer-atoi/submissions/
int myAtoi(string s);

int num_to_string(string s);

// using dynamic programming for Fibonacci
size_t fibonacciDP(size_t n);

// return number of ways to climb n stairs when you can either climb 1 or 2 steps each time.
int climbStairs(int n);

// given an array of positive integers called nums and a positive target integer, check if we can sum up to target using our array of integers
// using TABULATION 
bool canSum(int target, vector<int> nums);

// given an array of positive integers and a positive target integer, return an array of integers from nums that sums up to target
// using TABULATION 
vector<int> howSum(int target, vector<int> nums);

// determine if we can construct the string "target" from an array of strings
// using MEMOIZATION 
bool canConstruct(string target, vector<string> substrings);
bool constructUtil(string target, vector<string>& substrings, unordered_map<string, bool>& memo);
// using TABULATION
bool canConstruct2(string target, vector<string> substrings);


// determine if we can construct the string "target" from an array of strings
// using MEMOIZATION 
int waysConstruct(string target, vector<string> substrings);
int waysUtil(string target, vector<string>& substrings, unordered_map<string, int>& memo);
// using TABULATION
int waysConstruct2(string target, vector<string> substrings);


// determine all the combinations which we can construct the string "target" from an array of strings
// and return the 2d vector containing our combinations of substrings
// using MEMOIZATION 
vector<vector<string>> allConstruct(string target, vector<string> substrings);
vector<vector<string>> allWaysUtil(string target,
    vector<string>& substrings, unordered_map<string, vector<vector<string>>>& memo);
// using TABULATION
vector<vector<string>> allConstruct2(string target, vector<string> substrings);



// https://leetcode.com/problems/house-robber/submissions/
int rob(vector<int>& nums);

// https://leetcode.com/problems/house-robber-ii/submissions/
int rob2(vector<int>& nums);
int rob2Util(vector<int>& nums, int pos1, int pos2);

// https://leetcode.com/problems/decode-ways/submissions/
// Tabulation Dynamic Programming O(n) time & O(n) space
int numDecodings(string s);

// https://leetcode.com/problems/decode-ways/submissions/
// Dynamic Programming O(n) time & O(1) space
int numDecodings2(string s);

// https://leetcode.com/problems/unique-paths/
int uniquePaths(int m, int n);

// https://leetcode.com/problems/jump-game-iii/submissions/
bool canReach(vector<int>& arr, int start); 

// https://leetcode.com/problems/jump-game
bool canJump(vector<int>& nums);

// https://leetcode.com/problems/jump-game-ii
int jump(vector<int>& nums);

// https://leetcode.com/problems/longest-common-subsequence
int longestCommonSubsequence(string text1, string text2);

struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

// https://leetcode.com/problems/merge-k-sorted-lists/submissions/
ListNode* mergeKLists(vector<ListNode*>& lists);

// https://leetcode.com/problems/best-time-to-buy-and-sell-stock/submissions/
int maxProfit(vector<int>& prices);


// https://leetcode.com/problems/contains-duplicate/submissions/
bool containsDuplicate(vector<int>& nums);

//https://leetcode.com/problems/contains-duplicate-ii/submissions/
bool containsDuplicate2(vector<int>& nums, int k);

// https://leetcode.com/problems/product-of-array-except-self/submissions/
vector<int> productExceptSelf(vector<int>& nums);

// https://leetcode.com/problems/maximum-subarray/
int maxSubArray(vector<int>& nums);

// https://leetcode.com/problems/maximum-product-subarray/
int maxProduct(vector<int>& nums);

// https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
int findMin(vector<int>& nums);





