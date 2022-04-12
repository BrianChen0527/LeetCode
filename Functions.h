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

int longestConsecutive(vector<int>& nums);

int reverse(int x);

int myAtoi(string s);

int num_to_string(string s);

// using dynamic programming for Fibonacci
size_t fibonacciDP(size_t n);

// return number of ways to climb n stairs when you can either climb 1 or 2 steps each time.
int climbStairs(int n);

// determine if we can construct the string "target" from an array of strings
bool canConstruct(string target, vector<string> substrings);
bool constructUtil(string target, vector<string>& substrings, unordered_map<string, bool>& memo);


// determine if we can construct the string "target" from an array of strings
int waysConstruct(string target, vector<string> substrings);
int waysUtil(string target, vector<string>& substrings, unordered_map<string, bool>& memo);