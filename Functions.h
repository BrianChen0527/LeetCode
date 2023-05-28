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

// https://leetcode.com/problems/pacific-atlantic-water-flow/
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights);

//https://leetcode.com/problems/find-the-duplicate-number/description/
int findDuplicate(vector<int>& nums);

// https://leetcode.com/problems/spiral-matrix/description/
vector<int> spiralOrder(vector<vector<int>>& matrix);

// https://leetcode.com/problems/longest-increasing-subsequence/submissions/957462933/
int lengthOfLIS(vector<int>& nums);

// https://leetcode.com/problems/top-k-frequent-words/
vector<string> topKFrequent(vector<string>& words, int k);

// https://leetcode.com/problems/subsets/
void subsetsPermuter(vector<int>& nums, vector<vector<int>>& ans, vector<int>& path, int pos, int nums_size);
vector<vector<int>> subsets(vector<int>& nums);

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

int majorityElement(vector<int>& arr);

// https://leetcode.com/problems/word-break/
bool wordBreak(string s, vector<string>& wordDict);

// https://leetcode.com/problems/partition-equal-subset-sum/description/
bool canPartition(vector<int>& nums);

// https://leetcode.com/problems/number-of-1-bits/
int hammingWeight(uint32_t n);

// https://leetcode.com/problems/single-number/description/
int singleNumber(vector<int>& nums);

// https://leetcode.com/problems/accounts-merge/description/
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);

// https://leetcode.com/problems/move-zeroes/description/
void moveZeroes(vector<int>& nums);

// https://leetcode.com/problems/longest-palindrome/submissions/
int longestPalindromeLen(string s);

// https://leetcode.com/problems/product-of-array-except-self/
vector<int> productExceptSelf(vector<int>& nums);

// https://leetcode.com/problems/number-of-islands/
void islandBFS(vector<vector<char>>& grid, vector<pair<int, int>>& directions, int r, int c, int rows, int cols);
int numIslands(vector<vector<char>>& grid);

// https://leetcode.com/problems/rotting-oranges/
int orangesRotting(vector<vector<int>>& grid);

// https://leetcode.com/problems/search-in-rotated-sorted-array/
int searchRotatedSortedArray(vector<int>& nums, int target);

struct Node {
    int val;
    Node* next;
    Node() : val{ 0 }, next{ nullptr } {}
    Node(int x) : val{ x }, next{ nullptr } {}
    Node(int x, Node* next_in) : val{ x }, next{ next_in } {}
};

// https://leetcode.com/problems/course-schedule/
bool canFinish(int numCourses, vector<vector<int>>& prerequisites);

// https://leetcode.com/problems/evaluate-reverse-polish-notation/
int evalRPN(vector<string>& tokens);

// Write a program that reverses this singly-linked list of nodes
Node* reverse_list(Node* head);

// Write a function that deletes all duplicates in the list so that 
// each element ends up only appearing once.
Node* remove_duplicates(Node* head);

class GraphNode {
public:
    int val;
    vector<GraphNode*> neighbors;
    GraphNode() {
        val = 0;
        neighbors = vector<GraphNode*>();
    }
    GraphNode(int _val) {
        val = _val;
        neighbors = vector<GraphNode*>();
    }
    GraphNode(int _val, vector<GraphNode*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

// https://leetcode.com/problems/clone-graph/
GraphNode* cloneGraph(GraphNode* node);

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

// https://umich.instructure.com/courses/491173/files/folder/Exam%20-%20Practice/Exam%202%20-%20Final?preview=24280664
int number_of_tilings(int n);

int longestConsecutive(vector<int>& nums);

// file:///C:/Users/brian/Downloads/practice_final_2_combined_answers.pdf
bool zero_contiguous_sum(vector<int>& nums);

// file:///C:/Users/brian/Downloads/practice_final_2_combined_answers.pdf
struct Query { unsigned int id, start, end; 

Query(unsigned int i1, unsigned int s1, unsigned int e1) : id(i1), start(s1), end(e1) {}
};
void range_queries(const vector<unsigned int>& data,
    const vector<Query>& queries);

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

// https://leetcode.com/problems/word-break-ii/submissions/
// word berak 2
vector<string> allConstruct3(string target, vector<string> substrings);

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

// https://leetcode.com/problems/odd-even-linked-list/
ListNode* oddEvenList(ListNode* head);

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

// https://leetcode.com/problems/coin-change/
int coinChange(vector<int>& coins, int amount);

// https://leetcode.com/problems/search-in-rotated-sorted-array/submissions/
int search(vector<int>& nums, int target);

// https://leetcode.com/problems/merge-intervals/submissions/
vector<vector<int>> merge(vector<vector<int>>& intervals);

// https://leetcode.com/problems/insert-interval/submissions/
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval);

// https://leetcode.com/problems/reverse-linked-list/
ListNode* reverseList(ListNode* head);

// https://leetcode.com/problems/linked-list-cycle/
bool hasCycle(ListNode* head);

// https://leetcode.com/problems/merge-two-sorted-lists/
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2);

// https://leetcode.com/problems/longest-repeating-character-replacement/
int characterReplacement(string s, int k);

// https://leetcode.com/problems/minimum-height-trees/description/
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges);

// https://leetcode.com/problems/minimum-window-substring/
string minWindow(string s, string t);

// https://leetcode.com/problems/word-search/description/
bool wordSearch(vector<vector<char>>& board, vector<pair<short, short>>& dirs, string word, short r, short c, short pos, short rows, short cols);
bool exist(vector<vector<char>>& board, string word);

// https://leetcode.com/problems/word-break/submissions/
bool wordBreak(string s, vector<string>& wordDict);

// https://leetcode.com/problems/container-with-most-water/
int maxArea(vector<int>& height);

// https://leetcode.com/problems/letter-combinations-of-a-phone-number/
vector<string> letterCombinations(string digits);

// https://leetcode.com/problems/combination-sum-iv/
int combinationSum4(vector<int>& nums, int target);
int combinationHelper(vector<int>& nums, int target, unordered_map<int, int>& table);

// https://leetcode.com/problems/missing-number/solutions/?orderBy=most_votes&languageTags=cpp
int missingNumber(vector<int>& nums);

// https://leetcode.com/problems/group-anagrams/
vector<vector<string>> groupAnagrams(vector<string>& strs);
string countingSort(string s);

// https://leetcode.com/problems/missing-number/solutions/?orderBy=most_votes&languageTags=cpp
int missingNumber(vector<int>& nums);

// https://leetcode.com/problems/reverse-bits/
uint32_t reverseBits(uint32_t n);

// https://leetcode.com/problems/3sum/
vector<vector<int>> threeSum(vector<int>& nums);

// https://leetcode.com/problems/3sum/
vector<vector<int>> threeSum(vector<int>& nums);

// https://leetcode.com/problems/squares-of-a-sorted-array/
vector<int> sortedSquares(vector<int>& nums);

// https://leetcode.com/problems/01-matrix/
vector<vector<int>> updateMatrix(vector<vector<int>>& mat);

// https://leetcode.com/problems/01-matrix/
vector<vector<int>> updateMatrix(vector<vector<int>>&mat);

// https://leetcode.com/problems/k-closest-points-to-origin/
vector<vector<int>> kClosest(vector<vector<int>>& points, int k);

// https://leetcode.com/problems/longest-substring-without-repeating-characters/
int lengthOfLongestSubstring(string s);

// https://leetcode.com/problems/daily-temperatures/description/
vector<int> dailyTemperatures(vector<int>& temperatures);

// https://leetcode.com/problems/gas-station/description/
int canCompleteCircuit(vector<int>& gas, vector<int>& cost);


struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

// https://leetcode.com/problems/kth-smallest-element-in-a-bst/
int kthSmallest(TreeNode* root, int k);

// https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
TreeNode* treeBuilder(vector<int>& preorder, vector<int>& inorder,
    unordered_map<int, int>& inorder_mp, int i, int pl, int pr);
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);

// https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);

// https://leetcode.com/problems/binary-tree-level-order-traversal/
vector<vector<int>> levelOrder(TreeNode* root);

// https://leetcode.com/problems/subtree-of-another-tree/
bool isSubtree(TreeNode* root, TreeNode* subRoot);

// https://leetcode.com/problems/symmetric-tree/
bool isSymmetric(TreeNode* root);

// https://leetcode.com/problems/symmetric-tree/submissions/944700866/
bool compareNodes(TreeNode* l, TreeNode* r);
bool isSymmetric(TreeNode* root);

// https://leetcode.com/problems/invert-binary-tree/
TreeNode* invertTree(TreeNode* root);/**/

// https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
TreeNode* growTree(vector<int>& nums, int left, int right);
TreeNode* sortedArrayToBST(vector<int>& nums);

int maxSubarray(vector<int>& v);

int maxSubarrayProduct(vector<int>& v);

// https://leetcode.com/problems/task-scheduler/
int leastInterval(vector<char>& tasks, int n);










