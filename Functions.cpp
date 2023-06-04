using namespace std;
#include "Functions.h"
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


// https://leetcode.com/problems/letter-combinations-of-a-phone-number/
vector<string> letterCombinations(string digits) {
    unordered_map<char, string> mp{{'2', "abc"}, {'3', "def"}, {'4', "ghi"}, 
        {'5', "jkl"}, {'6', "mno"}, {'7', "pqrs"}, {'8', "tuv"}, {'9', "wxyz"}};

    queue<string> words;

    for (char d : digits) {
        if (d == '1') continue;
        int num_words = words.size();
        if (num_words == 0)
            for (char c : mp[d]) {
                string s;
                words.push(s + c);
            }
        for (; num_words > 0; num_words--) {
            string word = words.front();
            words.pop();

            for (char c : mp[d]) {
                words.push(word + c);
            }
        }
    }

    unordered_set<string> final_words;
    vector<string> ans;
    while (!words.empty()) {
        string word = words.front();

        if (final_words.find(word) == final_words.end()) ans.push_back(word);
        final_words.insert(word);
        words.pop();
    }
    return ans;
}


// https://leetcode.com/problems/word-search/description/
bool wordSearch(vector<vector<char>>& board, string word, short r, short c, short pos) {
    if (r < 0 || r >= board.size() || c < 0 || c >= board[0].size()) return false;
    if (board[r][c] != word[pos]) return false;
    if (pos == word.size() - 1) return true;

    char tmp = board[r][c];
    board[r][c] = '_';
    if (wordSearch(board, word, r + 1, c, pos + 1) ||
        wordSearch(board, word, r - 1, c, pos + 1) || 
        wordSearch(board, word, r, c - 1, pos + 1) || 
        wordSearch(board, word, r, c + 1, pos + 1)) return true;
    board[r][c] = tmp;
    return false;
};
bool exist(vector<vector<char>>& board, string word) {
    uint8_t len = word.size();
    int rows = board.size(), cols = board[0].size();
    for (short i = 0; i < rows; i++) {
        for (short j = 0; j < cols; j++) {
            if (board[i][j] == word[0])
                if (wordSearch(board, word, i, j, 0)) return true;
        }
    }
    return false;
}


// https://leetcode.com/problems/minimum-height-trees/description/
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 0) return {};
    if (n == 1) return { 0 };

    vector<vector<int>> adj(n, vector<int>());
    vector<int> degree(n, 0);

    for (int i = 0; i < n - 1; i++) {
        int n1 = edges[i][0], n2 = edges[i][1];
        adj[n1].push_back(n2);
        adj[n2].push_back(n1);
        degree[n1]++; degree[n2]++;
    }

    queue<int> leaves;
    vector<int> ans;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) {
            leaves.push(i);
            ans.push_back(i);
        }
    }

    while (n > 2) {
        ans.clear();
        int num_leaves = leaves.size();
        n -= num_leaves;
        for (int i = 0; i < num_leaves; i++) {
            int leaf = leaves.front(); 
            leaves.pop();
            degree[leaf]--;

            for (int node : adj[leaf]) {
                degree[node]--;
                if (degree[node] == 1) {
                    cout << node << endl;

                    leaves.push(node);
                    ans.push_back(node);
                }
            }
        }
    }
    return ans;
}


// https://leetcode.com/problems/find-all-anagrams-in-a-string/description/
vector<int> findAnagrams(string s, string p) {
    int plen = p.length(), slen = s.length();
    if (plen > slen) return {};

    vector<int> ms(26, 0), mp(26, 0); // ms is our sliding window
    for (int i = 0; i < plen; i++) {
        ms[s[i] - 'a']++;
        mp[p[i] - 'a']++;
    }

    vector<int> ans;
    if (ms == mp) ans.push_back(0);

    for (int i = plen; i < slen; i++) {
        ms[s[i - plen] - 'a']--;
        ms[s[i] - 'a']++;
        if (ms == mp) ans.push_back(i - plen + 1);
    }
    return ans;
}


// https://leetcode.com/problems/container-with-most-water/
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1;
    int maxA = 0;
    while (l < r) {
        int heightL = height[l], heightR = height[r];
        maxA = max(maxA, (r - l) * min(heightL, heightR));
        if (heightL < heightR) l++;
        else r--;
    }
    return maxA;
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

// Given a string s which consists of lowercase or uppercase letters, 
// return the length of the longest palindrome that can be built with those letters.
int longestPalindromeLen(string s) {
    int arr[58] = { 0 };
    for (char c : s) {
        arr[c - 'A']++;
    }

    int maxLen = 0;
    bool hasOdd = false;
    for (int i : arr) {
        if (i % 2 == 0){
            maxLen += i;
        }
        else {
            maxLen += (i - 1);
            hasOdd = true;
        }
    }
    return (hasOdd ? maxLen + 1 : maxLen);
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


// Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
bool compareNodes(TreeNode* l, TreeNode* r) {
    if (l && r && l->val != r->val) return false;
    else if ((!l && r) || (!r && l)) return false;
    else if (!l && !r) return true;
    else return compareNodes(l->left, r->right) && compareNodes(l->right, r->left);
}
bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return compareNodes(root->left, root->right);
}


// https://leetcode.com/problems/missing-number/solutions/?orderBy=most_votes&languageTags=cpp
int missingNumber(vector<int>& nums) {
    int missing = nums.size();
    for (int i = 0; i < nums.size(); i++) {
        missing ^= i;
        missing ^= nums[i];
    }
    return missing;
}

// https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
TreeNode* growTree(vector<int>& nums, int left, int right) {
    if (left == right) return new TreeNode(nums[right]);

    int mid = (left + right)/ 2;
    TreeNode* node = new TreeNode(nums[mid]);
    
    if(left != mid) node->left = growTree(nums, left, mid - 1);
    node->right = growTree(nums, mid + 1, right);
    return node;
}
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return growTree(nums, 0, nums.size() - 1);
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

// Given a string s, return the longest palindromic substring in s.
string longestPalindrome(string s) {
    int l = 0, r = 0, ptr = 0;
    int maxL = 0, maxR = 0;
    while (ptr < s.length()) {
        l = ptr;
        r = ptr;
        cout << ptr << endl;
        while (r + 1 < s.length() && s[r] == s[r + 1]) {
            r += 1;
            cout << l << " - " << r << endl;
        }
        while (l - 1 >= 0 && r + 1 < s.length() && s[l - 1] == s[r + 1]) {
            l--;
            r++;
            cout << l << " -- " << r << endl;
        }
        if (maxR - maxL < r - l) {
            maxR = r, maxL = l;
        }
        ptr++;
    }
    return s.substr(maxL, maxR - maxL + 1);
}

// https://leetcode.com/problems/binary-tree-level-order-traversal/
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> levels = {};
    queue<TreeNode*> level;
    level.push(root);

    while (!level.empty()) {
        int ptr = 0, num_nodes = level.size();
        queue<TreeNode*> nextLevel;
        vector<int> curr_level;

        while (ptr++ < num_nodes) {
            TreeNode* node = level.front();
            level.pop();

            if (!node) continue;

            curr_level.push_back(node->val);
            level.push(node->left);
            level.push(node->right);
        }
        if (!curr_level.empty()) levels.push_back(curr_level);
    }
    return levels;
}



// https://leetcode.com/problems/asteroid-collision/
vector<int> asteroidCollision(vector<int>& asteroids) {
    vector<int> result;

    for (int i = 0; i < asteroids.size(); i++) {
        if (asteroids[i] < 0) {
            while (!result.empty() && result.back() > 0 && result.back() < abs(asteroids[i])) {
                result.pop_back();
            }
            if (result.empty() || result.back() < 0)
                result.push_back(asteroids[i]);
            else if (result.back() == abs(asteroids[i]))
                result.pop_back();
        }
        else { 
            result.push_back(asteroids[i]);
        }
    } 
    return result;
}


// Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
// https://leetcode.com/problems/subarray-sum-equals-k/
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> sums;
    int sum = 0;
    int count = 0;

    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
        cout << sum << endl;

        if (sum == k)
            count++;

        if (sums.find(sum - k) != sums.end())
            count += sums[sum - k];

        sums.find(sum) == sums.end() ? sums[sum] = 1 : sums[sum]++;
    }
    return count;
}

// Given a non-empty array of integers nums, every element appears twice except for one. Find it.
// You must implement a solution with a linear runtime complexityand use only constant extra space.
int singleNumber(vector<int>& nums) {
    return accumulate(nums.begin(), nums.end(), 0, [](int a, int b) { return a ^ b; });
}


// Given an integer array nums, move all 0's to the end of it while maintaining the 
// relative order of the non-zero elements.
void moveZeroes(vector<int>& nums) {
    int j = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            swap(nums[j], nums[i]);
            j++;
        }
    }
}


int majorityElement(vector<int>& arr) {
    int ele = arr[0];
    int count = 0;
    for (int i = 0; i < arr.size(); i++) {
        if (count == 0)ele = arr[i];
        count += (ele == arr[i]) ? 1 : -1;
    }
    return ele;
}

// Write a function that takes the binary representation of an unsigned 
// integer and returns the number of '1' bits it has (Hamming weight).
int hammingWeight(uint32_t n) {
    int ones = 0;
    while (n) {
        if (n & 1) {
            ones++;
        }
        n >>= 1;
    }
    return ones;
}


// Given an unsorted array of integers nums, 
// return the length of the longest consecutive elements sequence.
//https://leetcode.com/problems/longest-consecutive-sequence/
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numbers;
    for (auto c : nums) {
        numbers.insert(c);
    }
    int maxCount = 0;
    for (auto it : numbers) {
        int low = it - 1, high = it + 1, count = 1;

        while (numbers.find(low) != numbers.end()) {
            numbers.erase(low);
            low--; count++;
        }
        while (numbers.find(high) != numbers.end()) {
            numbers.erase(high);
            high++; count++;
        }
        maxCount = max(maxCount, count);
    }
    return maxCount;
}

int reverse(int x) {
    string s = to_string(x);
    bool neg = x < 0;
    if (neg) s = s.substr(1);

    for (int i = 0; i < s.length() / 2; i++) {
        cout << s[i] << " " << s[s.length() - i - 1] << endl;
        swap(s[i], s[s.length() - i - 1]);
    }

    if (s.length() > 10) return 0;
    if (s.length() == 10 && ((s.compare("2147483647") > 0 && !neg) || (s.compare("2147483648") > 0 && neg)))
        return 0;

    x = stoi(s);
    return (neg ? x * -1 : x);
}


// https://leetcode.com/problems/daily-temperatures/description/
vector<int> dailyTemperatures(vector<int>& temperatures) {
    stack<int> highest_temps_idx = {};
    vector<int> ans(temperatures.size(), 0);
    for (int i = temperatures.size() - 1; i >= 0; i--) {
        int temp = temperatures[i];
        while (!highest_temps_idx.empty() && temp >= temperatures[highest_temps_idx.top()]) {
            highest_temps_idx.pop();
        }
        if (!highest_temps_idx.empty()) ans[i] = highest_temps_idx.top() - i;

        highest_temps_idx.push(i);
    }
    return ans;
}


// https://leetcode.com/problems/top-k-frequent-words/
vector<string> topKFrequent(vector<string>& words, int k) {
    unordered_map<string, int> counter;

    for (auto w : words) counter[w]++;

    auto comp = [&](const pair<int, string>& p1, const pair<int, string>& p2) {
        return (p1.first == p2.first ? p1.second > p2.second : p1.first < p2.first); };
    priority_queue < pair<int, string>, vector<pair<int, string>>, decltype(comp) > Q(comp);
    for (auto k : counter) Q.emplace(k.second, k.first);

    vector<string> ans;
    for (int i = 0; i < k; i++) {
        ans.push_back(Q.top().second);
        Q.pop();
    }
    return ans;
}


// https://leetcode.com/problems/course-schedule-ii/
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> adj(numCourses, vector<int>());
    vector<int> degrees(numCourses, 0);

    for (auto v : prerequisites) {
        adj[v[1]].push_back(v[0]);
        degrees[v[0]]++;
    }

    vector<int> schedule; schedule.reserve(numCourses);
    queue<int> canTake;
    for (int j = 0; j < numCourses; j++) {
        if (degrees[j] == 0) {
            degrees[j]--;
            canTake.push(j);
            schedule.push_back(j);
        }
    }

    while (!canTake.empty()) {
        int curr = canTake.front();
        for (auto c : adj[curr]) {
            if (--degrees[c] == 0) {
                canTake.push(c);
                schedule.push_back(c);
            }
        }
        canTake.pop();
    }
    return schedule.size() == numCourses ? schedule : vector<int>();
}


// https://leetcode.com/problems/longest-increasing-subsequence/submissions/957462933/
int lengthOfLIS(vector<int>& nums) {
    vector<int> ans = { nums[0] };

    for (int i = 1; i < nums.size(); i++) {
        auto it = lower_bound(ans.begin(), ans.end(), nums[i]);
        if (it != ans.end()) ans.erase(it);
        
        ans.insert(it, nums[i]);
    }
    return ans.size();
}


// https://leetcode.com/problems/kth-smallest-element-in-a-bst/
int kthSmallest(TreeNode* root, int k) {
    stack<TreeNode*> dfs;
    int pos = 0;
    while (pos < k) {
        while (root) {
            dfs.push(root);
            root = root->left;
        }
        root = dfs.top();
        if (++pos == k) return root->val;
        dfs.pop();
        root = root->right;
    }
    return -1;
}


//https://leetcode.com/problems/find-the-duplicate-number/description/
int findDuplicate(vector<int>& nums) {
    int slow = nums[0], fast = nums[nums[0]];

    while (slow != fast) {
        slow = nums[slow];
        fast = nums[nums[fast]];
    }

    fast = nums[nums[0]];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[nums[fast]];
    }
    return fast;
}


// https://leetcode.com/problems/pacific-atlantic-water-flow/
void waterFlowDFS(vector<vector<int>>& heights, vector<vector<short>>& land, int prev_height, int r, int c, int rows, int cols) {
    land[r][c] = 1;
    int currH = heights[r][c];
    if (r + 1 < rows && heights[r + 1][c] >= currH && land[r + 1][c] != 1) waterFlowDFS(heights, land, heights[r][c], r + 1, c, rows, cols);
    if (c - 1 >= 0 && heights[r][c - 1] >= currH && land[r][c - 1] != 1) waterFlowDFS(heights, land, heights[r][c], r, c - 1, rows, cols);
    if (r - 1 >= 0 && heights[r - 1][c] >= currH && land[r - 1][c] != 1) waterFlowDFS(heights, land, heights[r][c], r - 1, c, rows, cols);
    if (c + 1 < cols && heights[r][c + 1] >= currH && land[r][c + 1] != 1) waterFlowDFS(heights, land, heights[r][c], r, c + 1, rows, cols);
}
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    int rows = heights.size(), cols = heights[0].size();
    vector<vector<short>> atl(rows, vector<short>(cols, 0));
    vector<vector<short>> pac(rows, vector<short>(cols, 0));

    for (int i = 0; i < rows; i++) {
        pac[i][0] = 1;
        atl[i][cols - 1] = 1;
    }

    pac[0] = vector<short>(cols, 1);
    atl[rows - 1] = vector<short>(cols, 1);
    atl[0][cols - 1] = atl[rows - 1][0] = pac[0][cols - 1] = pac[rows - 1][0] = 1;
    
    for (int i = 0; i < rows; i++) {
        waterFlowDFS(heights, pac, heights[i][0], i, 0, rows, cols);
        waterFlowDFS(heights, atl, heights[i][cols - 1], i, cols - 1, rows, cols);
    }
    for (int j = 0; j < cols; j++) {
        waterFlowDFS(heights, pac, heights[0][j], 0, j, rows, cols);
        waterFlowDFS(heights, atl, heights[rows - 1][j], rows - 1, j, rows, cols);
    }
    vector<vector<int>> ans;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (pac[i][j] == 1 && atl[i][j] == 1) ans.push_back({ i, j });
        }
    }

    //for (int i = 0; i < rows; i++) {
    //    for (int j = 0; j < cols; j++) {
    //        cout << pac[i][j] << "\t";
    //    }
    //    cout << endl;
    //}
    return ans;
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


// parses string, and extracts integer from string (there can only be one continuous number in string)

/*
int advancedAtoi(string s) {
    if (s.empty()) return 0;

    int pos = 0, neg = 0;
    string num = "";
    while (s[pos] != '-' && (s[pos] - '0' < 0 || s[pos] - '9' > 0) && pos < s.length()) { pos++; }
    if (s[pos] == '-') {
        neg = 1;
        pos++;
    }

    while (s[pos] - '0' => 0 && s[pos] - '9' <= 0 && pos < s.length()) {
        num += s[pos];
        pos++;
    }

    if (num.length() > 10) return neg ? INT32_MIN : INT32_MAX;
    else if (num.length() == 10 && neg && num.compare("2147483648") > 0) return INT32_MIN;
    else if (num.length() == 10 && !neg && num.compare("2147483647") > 0) return INT32_MAX;

    return neg ? -1 * stoi(num) : stoi(num);
}
*/
// https://leetcode.com/problems/string-to-integer-atoi/submissions/
int myAtoi(string s) {
    if (s.empty()) return 0;

    int pos = 0, neg = 0;
    while (s[pos] == ' ') pos++;
    if (s[pos] != '-' && s[pos] != '+' && (s[pos] - '0' < 0 || s[pos] - '9' > 0)) return 0;

    long num = 0;
    if (s[pos] == '-') { neg = 1; pos++; }
    else if (s[pos] == '+') pos++;

    while (s[pos] - '0' >= 0 && s[pos] - '9' <= 0 && num < INT32_MAX)
        num = num * 10 + (s[pos++] - '0');

    neg ? num *= -1 : num = num;
    if (num > INT32_MAX) return INT32_MAX;
    else if (num < INT32_MIN) return INT32_MIN;
    else return num;
}


int fibonacci(int n) {
    if (n == 1 || n == 0) return 1;
    else return fibonacci(n - 1) + fibonacci(n - 2);
}

size_t fibonacciDP(size_t n) {
    if (n < 2) return n;

    vector<size_t> fib(n + 1);
    fib[0] = 0; fib[1] = 1;
    for (int i = 2; i < n + 1; i++)
        fib[i] = fib[i - 1] + fib[i - 2];

    return fib[n];
}


// https://leetcode.com/problems/permutations/
void permuteHelper(vector<int>& nums, vector<int>& path, vector<vector<int>>& ans, unordered_set<int>& visited, int len, int tot_size){
    
    for (int n : nums) {
        if (visited.find(n) != visited.end()) continue;
        path[len] = n;

        if (len == tot_size - 1) {
            ans.push_back(path);
            return;
        }
        
        visited.insert(n);
        permuteHelper(nums, path, ans, visited, len + 1, tot_size);
        visited.erase(n);
    }
}

vector<vector<int>> permute(vector<int>& nums) {
    unordered_set<int> visited;
    int tot_size = nums.size();
    vector<int> path (tot_size, -1);
    vector<vector<int>> ans;
    permuteHelper(nums, path, ans, visited, 0, tot_size);
    return ans;
}


int num_to_string(string s) {
    int iter1 = 1, sum = 1, size = s.length();
    vector<int> sections;

    // Invalid cases
    if (s[0] == '0') return 0;
    if (s.length() == 1) return 1;

    for (int i = 0; i < size - 1; i++) {
        // check for double 0s
        if (s[i] == '0' && s[i + 1] == '0')
            return 0;

        // check for 0s
        if (s[i] == '0' && i != 0) {
            sections.push_back(iter1 - 1);
            iter1 = 0;
        }

        int num = stoi(s.substr(i, 2));
        // check for number greater than 26
        if (num > 26) {
            // check for invalid case of X0 where X is larger than 2
            if (num % 10 == 0)
                return 0;
            sections.push_back(iter1);
            iter1 = 0;
        }
        iter1++;
    }
    // if last digit is 0, last two digits only provide 1 permutation
    if (s[size - 1] == '0') iter1 -= 2;
    sections.push_back(iter1);

    for (int i : sections) sum *= fibonacciDP(i);
    return sum;
}


// return number of ways to climb n stairs when you can either climb 1 or 2 steps each time.
int climbStairs(int n) {
    vector<int> solutions(n + 1, 0);

    solutions[0] = 0;
    solutions[1] = 1;
    for (int i = 2; i <= n; i++) {
        solutions[i] = solutions[i - 1] + solutions[i - 2];
    }
    return solutions[n];
}

bool canSum(int target, vector<int> nums) {
    vector<bool> tabulation(target + 1, false);
    tabulation[0] = true;

    for (int j = 0; j < target + 1; j++)
        for (int i : nums)
            if (tabulation[j] && j + i <= target)
                tabulation[j + i] = true;

    for (int i = 0; i < tabulation.size(); i++) {
        cout << i << ":  " << tabulation[i] << endl;
    }

    return tabulation[target];
}


vector<int> howSum(int target, vector<int> nums) {
    vector<vector<int>> table(target + 1);
    table[0] = { 0 };

    for (int j = 0; j < target + 1; j++)
        for (int i : nums) {
            if (table[j].size() > 0 && j + i <= target) {
                if (j == 0) table[i + j] = { i };
                else {
                    table[i + j] = table[j];
                    table[i + j].push_back(i);
                }
            }
        }

    return table[target];
}

// given an array of integers nums and a target number, return the shortest combination of 
// integers in nums that sums up to target
vector<int> bestSum(int target, vector<int> nums) {
    vector<vector<int>> table(target + 1);
    table[0] = { 0 };

    for (int j = 0; j < target + 1; j++)
        for (int i : nums) {
            if (table[j].size() > 0 && j + i <= target) {
                if (j == 0) table[i + j] = { i };
                else if (table[i + j].empty() || table[i + j].size() > table[j].size() + 1) {
                    table[i + j] = table[j];
                    table[i + j].push_back(i);
                }
            }
        }
    return table[target];
}


// https://leetcode.com/problems/next-permutation/
void nextPermutation(vector<int>& nums) {
    int i = nums.size() - 2;
    for (; i >= 0; --i) {
        if (nums[i] < nums[i + 1]) break;
    }
    if (i < 0) {
        reverse(nums.begin(), nums.end());
    }
    else {
        int j = nums.size() - 1;
        while (nums[j] <= nums[i])
            j--;
        swap(nums[j], nums[i]);
        reverse(nums.begin() + i + 1, nums.end());
    }
}

// https://leetcode.com/problems/gas-station/description/
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int n = gas.size();
    int min_gas = 1, min_pos = 0, curr_cost = 0;

    for (int i = 0; i < n * 2; i++) {
        int j = i % n;
        curr_cost += (gas[j] - cost[j]);
        if (curr_cost < min_gas) {
            min_gas = curr_cost;
            min_pos = i;
        }
    }
    return curr_cost < 0 ? -1 : (min_pos + 1);
}

// determine if we can construct the string "target" from an array of strings
// using MEMOIZATION 
bool canConstruct(string target, vector<string> substrings) {
    unordered_map<string, bool> memo;
    return constructUtil(target, substrings, memo);
}

// Helper function
bool constructUtil(string target, vector<string>& substrings, unordered_map<string, bool>& memo) {
    if (target.empty()) return true;
    if (memo.find(target) != memo.end()) return memo[target];

    for (string s : substrings) {
        if (target.length() >= s.length() && target.substr(0, s.length()) == s) {
            string sub = target.substr(s.length());

            if (constructUtil(sub, substrings, memo)) {
                memo[target] = true;
                return true;
            }
        }
    }
    memo[target] = false;
    return false;
}


// https://leetcode.com/problems/find-k-closest-elements/
vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    int pos = lower_bound(arr.begin(), arr.end(), x) - arr.begin();
    int low = pos - 1, high = pos, len = arr.size();

    deque<int> nums;
    while (low >= 0 && high < len && k > 0) {
        if (x - arr[low] <= arr[high] - x) {
            nums.push_front(arr[low--]);
        }
        else {
            nums.push_back(arr[high++]);
        }
        k--;
    }

    while (low >= 0 && k > 0) {
        nums.push_front(arr[low--]);
        k--;
    }

    while (high < len && k > 0) {
        nums.push_back(arr[high++]);
        k--;
    }
    return vector<int>(nums.begin(), nums.end());
}


// determine if we can construct the string "target" from an array of strings
// using TABULATION
bool canConstruct2(string target, vector<string> substrings) {
    vector<bool> table(target.length() + 1, false);
    table[0] = true;
    for (int i = 0; i < target.length(); i++) {
        for (string s : substrings) {
            if (target[i] == s[0] && table[i] && target.substr(i, s.length()) == s) {
                table[i + s.length()] = true;
            }
        }
    }
    return table[target.length()];
}



// determine the number of ways we can construct the string "target" from an array of strings
int waysConstruct(string target, vector<string> substrings) {
    unordered_map<string, int> memo;
    int count = 0;
    return waysUtil(target, substrings, memo);
}
// Helper function
int waysUtil(string target, vector<string>& substrings, unordered_map<string, int>& memo) {
    if (target.empty()) return 1;
    if (memo.find(target) != memo.end()) return memo[target];

    int totalWays = 0;
    for (string s : substrings) {
        if (target.length() >= s.length() && target.substr(0, s.length()) == s) {
            string sub = target.substr(s.length());
            totalWays += waysUtil(sub, substrings, memo);
        }
    }
    memo[target] = totalWays;
    return totalWays;
}

// determine the number of ways we can construct the string "target" from an array of strings
// using TABULATION
int waysConstruct2(string target, vector<string> substrings) {
    vector<int> table(target.length() + 1, 0);
    table[0] = 1;
    for (int i = 0; i < target.length(); i++) {
        for (string s : substrings) {
            if (target[i] == s[0] && table[i] && target.substr(i, s.length()) == s) {
                table[i + s.length()] += table[i];
            }
        }
    }
    return table[target.length()];
}



// determine all the combinations which we can construct the string "target" from an array of strings
// and return the 2d vector containing our combinations of substrings
vector<vector<string>> allConstruct(string target, vector<string> substrings) {
    unordered_map<string, vector<vector<string>>> memo;
    return allWaysUtil(target, substrings, memo);
}

// Helper function
vector<vector<string>> allWaysUtil(string target,
    vector<string>& substrings, unordered_map<string, vector<vector<string>>>& memo) {
    vector<vector<string>> v = { {} };
    if (target.empty()) return v;
    if (memo.find(target) != memo.end()) return memo[target];

    vector<vector<string>> allCombinations;
    for (string s : substrings) {
        if (target.length() >= s.length() && target.substr(0, s.length()) == s) {
            string sub = target.substr(s.length());
            for (vector<string>& v : allWaysUtil(sub, substrings, memo)) {
                v.insert(v.begin(), s);
                allCombinations.push_back(v);
            }
        }
    }
    memo[target] = allCombinations;
    return allCombinations;
}
// determine all the combinations which we can construct the string "target" from an array of strings
// and return the 2d vector containing our combinations of substrings
// using TABULATION
vector<vector<string>> allConstruct2(string target, vector<string> substrings) {
    vector<vector<vector<string>>> table(target.length() + 1, vector<vector<string>>{});
    table[0] = { {} };

    for (int i = 0; i < target.length(); i++) {
        for (string s : substrings) {
            if (target[i] == s[0] && !table[i].empty() && target.substr(i, s.length()) == s) {
                vector<vector<string>> vec = table[i];
                for (vector<string>& v : vec) {
                    v.push_back(s);
                    table[i + s.length()].push_back(v);
                }
            }
        }
    }
    return table[target.length()];
}

// https://leetcode.com/problems/word-break-ii/submissions/
// word berak 2
vector<string> allConstruct3(string target, vector<string> substrings) {
    vector<vector<string>> table(target.length() + 1);
    table[0] = {""};

    for (int i = 0; i < target.length(); i++) {
        for (string sub : substrings) {
            if (target[i] == sub[0] && !table[i].empty() && target.substr(i, sub.length()) == sub) {
                vector<string> vec = table[i];
                for (string str: vec) {
                    if (str.empty()) str = sub;
                    else str += (" " + sub);
                    table[i + sub.length()].push_back(str);
                }
            }
        }
    }
    return table[target.length()];
}



// https://leetcode.com/problems/house-robber/submissions/
int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    if (nums.size() == 1) return nums[0];
    if (nums.size() == 2) return max(nums[0], nums[1]);

    vector<int> table(nums.size() + 1, 0);
    table[1] = nums[0];
    table[2] = nums[1];
    for (int i = 3; i < table.size(); i++) {
        table[i] = max(table[i - 3], table[i - 2]) + nums[i - 1];
    }
    return max(table.at(table.size() - 2), table.back());
}

int rob2(vector<int>& nums) {
    if (nums.size() < 2)
        return nums.size() ? nums[0] : 0;
    return max(rob2Util(nums, 1, nums.size()), rob2Util(nums, 0, nums.size() - 1));
}

int rob2Util(vector<int>& nums, int pos1, int pos2) {
    if (nums.size() < 2)
        return nums.size() ? nums[0] : 0;

    int curr = 0, prev = 0;
    for (int i = pos1; i < pos2; i++) {
        int next = max(curr, prev + nums[i]);
        prev = curr;
        curr = next;
    }
    return curr;
}

// https://leetcode.com/problems/decode-ways/submissions/
// Tabulation Dynamic Programming O(n) time & O(n) space
int numDecodings(string s) {
    if (!s.length()) return 0;
    vector<int> table(s.length() + 1, 0);
    table.back() = 1;
    for (int i = s.length() - 1; i >= 0; i--) {
        if (s[i] != '0')
            table[i] += table[i + 1];
        if (i + 1 < s.length() && stoi(s.substr(i, 2)) <= 26 && s[i] != '0')
            table[i] += table[i + 2];
    }
    return table[0];
}

// https://leetcode.com/problems/decode-ways/submissions/
// Dynamic Programming O(n) time & O(1) space
int numDecodings2(string s) {
    if (!s.length()) return 0;

    int prev2 = 1, prev1 = 1;
    for (int i = s.length() - 1; i >= 0; i--) {
        int tmp = 0;
        if (s[i] != '0')
            tmp += prev1;
        if (i + 1 < s.length() && stoi(s.substr(i, 2)) <= 26 && s[i] != '0')
            tmp += prev2;
        prev2 = prev1;
        prev1 = tmp;
    }
    return prev1;
}


// https://leetcode.com/problems/unique-paths/
int uniquePaths(int m, int n) {
    vector<vector<int>> table(m, vector<int>(n, 1));
    for (int r = 1; r < m; r++) {
        for (int c = 1; c < n; c++) {
            table[r][c] = table[r - 1][c] + table[r][c - 1];
        }
    }
    return table[m - 1][n - 1];
}

// https://leetcode.com/problems/jump-game-iii/submissions/
bool canReach(vector<int>& arr, int start) {
    queue<int> q;
    q.push(start);
    vector<bool> visited(arr.size(), false);
    while (!q.empty()) {
        int pos = q.front(); q.pop();
        if (arr[pos] == 0) return true;
        visited[pos] = true;
        int front = pos + arr[pos], back = pos - arr[pos];
        if (front < arr.size() && !visited[front]) q.push(front);
        if (back >= 0 && !visited[back]) q.push(back);
    }
    return false;
}


// https://leetcode.com/problems/accounts-merge/description/
//void accountsDFS(unordered_map<string, vector<int>>& email_accounts, vector<vector<string>>& accounts, vector<bool>& visited, unordered_set<string>& emails, int account) {
//    if (visited[account]) return;
//    visited[account] = true;
//
//    for (int i = 1; i < accounts[account].size(); i++) {
//        string email = accounts[account][i];
//        if (emails.find(email) == emails.end()) emails.insert(email);
//        for (auto i : email_accounts[email]) {
//            accountsDFS(email_accounts, accounts, visited, emails, i);
//        }
//    }
//}


//vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
//    unordered_map<string, vector<int>> email_accounts;
//    vector<bool> visited(accounts.size(), false);
//
//    for (int j = 0; j < accounts.size(); j++) {
//        vector<string>& account = accounts[j];
//        for (int i = 1; i < accounts[j].size(); i++) {
//            email_accounts[account[i]].push_back(j);
//        }
//    }
//
//    vector<vector<string>> ans;
//    for (int i = 0; i < accounts.size(); i++) {
//        if (visited[i]) continue;
//        
//        unordered_set<string> emails;
//        accountsDFS(email_accounts, accounts, visited, emails, i);
//
//        int pos = 1;
//        ans.emplace_back(emails.size() + 1, accounts[i][0]);
//        for (auto e : emails) ans.back()[pos++] = e;
//        sort(ans.back().begin() + 1, ans.back().end());
//    }
//    return ans;
//}


// https://leetcode.com/problems/add-two-numbers/
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* head = new ListNode(0);
    ListNode* curr = head;
    bool carry = false;

    while (l1 && l2) {
        curr->next = new ListNode(0);
        curr = curr->next;
        int s = l1->val + l2->val + (carry ? 1 : 0);
        carry = (s >= 10);
        curr->val = s % 10;
        l1 = l1->next;
        l2 = l2->next;
    }
    while (l2) {
        curr->next = new ListNode(0);
        curr = curr->next;
        curr->val = l2->val + (carry ? 1 : 0);
        carry = false;

        if (curr->val > 9) {
            curr->val = 0;
            carry = true;
        }
        l2 = l2->next;
    }
    while (l1) {
        curr->next = new ListNode(0);
        curr = curr->next;
        curr->val = l1->val + (carry ? 1 : 0);
        carry = false;

        if (curr->val > 9) {
            curr->val = 0;
            carry = true;
        }
        l1 = l1->next;
    }

    if (carry) curr->next = new ListNode(1);
    return head;
}


// https://leetcode.com/problems/generate-parentheses/
void parenthesisPermutor(int n, int l, int r, string curr, vector<string>& ans) {
    if (l == n && r == n) { return ans.push_back(curr); }
    
    if (l < n) {
        parenthesisPermutor(n, l + 1, r, curr + "(", ans);
    }
    if (r < n && r < l) {
        parenthesisPermutor(n, l, r + 1, curr + ")", ans);
    }
}

vector<string> generateParenthesis(int n) {
    vector<string> ans;
    parenthesisPermutor(n, 0, 0, "", ans);
    return ans;
}


// https://leetcode.com/problems/sort-list/solutions/1795126/c-merge-sort-2-pointer-easy-to-understand/
ListNode* mergeList(ListNode* l, ListNode* r) {
    ListNode* tmp = new ListNode(0);
    ListNode* head = tmp;

    while (l && r) {
        if (l->val > r->val) {
            tmp->next = r;
            r = r->next;
        }
        else{
            tmp->next = l;
            l = l->next;
        }
        tmp = tmp->next;
    }
    tmp->next = r ? r : l;
    return head->next;
}

ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;
    
    ListNode* tmp = head;
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
        tmp = slow;
        slow = slow->next;
        fast = fast->next->next;
    }

    tmp->next = nullptr;

    ListNode* l = sortList(head);
    ListNode* r = sortList(slow);
    return mergeList(l, r);
}


// https://leetcode.com/problems/maximal-square/solutions/600149/python-thinking-process-diagrams-dp-approach/
int maximalSquare(vector<vector<char>>& matrix) {
    int rows = matrix.size(), cols = matrix[0].size();
    vector<vector<int>> dp(rows, vector<int>(cols, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dp[i][j] = matrix[i][j] - '0';
        }
    }

    int max_area = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 1; j < cols; j++) {
            max_area = max(max_area, dp[i][j]);
            
            int sqr = 0;
            if (j - 1 >= 0 && i - 1 >= 0) {
                sqr = min({dp[i][j - 1], dp[i - 1][j - 1], dp[i - 1][j]});

                if (dp[i][j] && sqr) {
                    dp[i][j] = ++sqr;
                    max_area = max(max_area, sqr);
                }
            }
        }
    }

    return pow(max_area, 2);
}


// https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/1019513/python-quickselect-average-o-n-explained/
int partition(vector<int>& nums, int l, int r) {
    if (l == r) return r;

    // use random index to speed up
    int rand_idx = l + rand() % (r - l);
    swap(nums[rand_idx], nums[r]);

    int i = l;
    for (int j = l; j < r; j++) {
        if (nums[j] < nums[r]) swap(nums[i++], nums[j]);
    }
    swap(nums[r], nums[i]);
    return i;
}
int quickSelect(vector<int>& nums, int l, int r, int k) {
    int pivot = partition(nums, l, r);

    if (k == r - pivot + 1) return nums[pivot];
    if (k <= r - pivot) return quickSelect(nums, pivot + 1, r, k);
    return quickSelect(nums, l, pivot - 1, k - (r - pivot + 1));
}

int findKthLargest(vector<int>& nums, int k) {
    srand(time(NULL));
    return quickSelect(nums, 0, nums.size() - 1, k);
}


// https://leetcode.com/problems/random-pick-with-weight/submissions/
class Solution {
public:
    vector<int> weights;
    int total;
    Solution(vector<int>& w) {
        srand(time(NULL));

        weights.resize(w.size());
        weights[0] = total = w[0];
        for (int i = 1; i < w.size(); i++) {
            weights[i] = w[i] + weights[i - 1];
            total += w[i];
        }
    }

    int pickIndex() {
        int rand_idx = rand() % total;
        return upper_bound(weights.begin(), weights.end(), rand_idx) - weights.begin();
    }
};


// https://leetcode.com/problems/jump-game
bool canJump(vector<int>& nums) {
    if (nums.size() < 2) return true;
    int maxPos = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] == 0 && maxPos < i) return false;
        maxPos = max(maxPos, i + nums[i]);
    }
    return maxPos >= nums.size() - 1;
}

// https://leetcode.com/problems/jump-game-ii
int jump(vector<int>& nums) {
    if (nums.size() < 2) return 0;
    int currMaxPos = 0, nextMaxPos = 0, i = 0, jumps = 0;

    while (currMaxPos < nums.size() - 1) {
        nextMaxPos = max(nextMaxPos, i + nums[i]);

        if (i == currMaxPos) {
            currMaxPos = nextMaxPos;
            jumps++;
        }

        i++;
    }
    return jumps;
}

// https://leetcode.com/problems/longest-common-subsequence
int longestCommonSubsequence(string text1, string text2) {
    vector<vector<int>> table(text1.length() + 1, vector<int>(text2.length() + 1, 0));

    for (int r = 1; r < table.size(); r++) {
        for (int c = 1; c < table.at(0).size(); c++) {
            if (text1[r - 1] == text2[c - 1]) table[r][c] = table[r - 1][c - 1] + 1;
            else table[r][c] = max(table[r][c - 1], table[r - 1][c]);
        }
    }

    return table[table.size() - 1][table.at(0).size() - 1];
}

// https://leetcode.com/problems/merge-k-sorted-lists/submissions/
ListNode* mergeKLists(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;
    deque<ListNode*> toMerge;
    for (ListNode* ptr : lists) {
        toMerge.push_back(ptr);
    }

    while (toMerge.size() > 1) {
        ListNode* ptr1 = toMerge.front(); toMerge.pop_front();
        ListNode* ptr2 = toMerge.front(); toMerge.pop_front();
        ListNode* curr = new ListNode(0, nullptr);
        ListNode* head = curr;

        while (ptr1 && ptr2) {
            if (ptr1->val > ptr2->val) {
                curr->next = ptr2;
                ptr2 = ptr2->next;
            }
            else {
                curr->next = ptr1;
                ptr1 = ptr1->next;
            }
            curr = curr->next;
        }
        if (ptr1)
            curr->next = ptr1;
        if (ptr2)
            curr->next = ptr2;
        head = head->next;
        toMerge.push_back(head);
    }
    return toMerge.front();
}

// https://leetcode.com/problems/best-time-to-buy-and-sell-stock/submissions/
int maxProfit(vector<int>& prices) {
    uint16_t minPrice = INT32_MAX, maxProfit = 0;
    for (uint16_t p : prices) {
        if (p - minPrice > maxProfit) maxProfit = p - minPrice;
        if (p < minPrice) minPrice = p;
    }
    return maxProfit;
}

// https://leetcode.com/problems/contains-duplicate/submissions/
bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> appeared;
    for (int n : nums) {
        if (appeared.find(n) != appeared.end())
            return true;
        appeared.insert(n);
    }
    return false;
}


// https://leetcode.com/problems/maximum-subarray/
int maxSubArray(vector<int>& nums) {
    int curr = nums[0], maxN = curr;
    for (int i = 1; i < nums.size(); i++) {
        if (curr < 0) {
            if (nums[i] > curr) {
                curr = nums[i];
                maxN = curr;
            }
        }
        else {
            if (nums[i] + curr > 0) {
                curr += nums[i];
                maxN = max(maxN, curr);
            }
            else curr = 0;
        }
    }
    return maxN;
}



// https://leetcode.com/problems/maximum-product-subarray/
int maxProduct(vector<int>& nums) {
    int curr_min = 1, curr_max = 1, all_time_max = INT_MIN;
    
    for (int n : nums) {
        if (n < 0) {
            int tmp_min = curr_min;
            curr_min = min(curr_max * n, n);
            curr_max = max(tmp_min * n, n);
        }
        else {
            curr_min = min(curr_min * n, n);
            curr_max = max(curr_max * n, n);
        }
        all_time_max = max(all_time_max, curr_max);
        curr_max = (curr_max == 0 ? 1 : curr_max);
        curr_min = (curr_min == 0 ? 1 : curr_min);
    }
    return all_time_max;
}

int maxProduct2(vector<int>& nums) {
    int minP = nums[0], maxP = minP, trueMax = maxP;
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] < 0) {
            swap(minP, maxP);
        }
        minP = min(minP * nums[i], nums[i]);
        maxP = max(maxP * nums[i], nums[i]);
        cout << minP << " " << maxP << endl;

        trueMax = max(maxP, trueMax);
        if (nums[i] == 0) {
            minP = 1; maxP = 1;
        }
    }
    return trueMax;
}


// https://leetcode.com/problems/search-in-rotated-sorted-array/submissions/
int search(vector<int>& nums, int target) {
    int left = 0, mid = nums.size() / 2, right = nums.size() - 1;
    while (right >= left) {
        int l = nums[left], r = nums[right], m = nums[mid];
        if (target == m) return mid;

        if (m > r)
            (target < m&& target >= l) ? right = mid - 1 : left = mid + 1;
        else if (l > m)
            (target > m && target <= r) ? left = mid + 1 : right = mid - 1;
        else
            (target > m) ? left = mid + 1 : right = mid - 1;
        mid = (left + right) / 2;
    }
    return -1;
}


// https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root->val == p->val || root->val == q->val) return root;
    TreeNode* l = lowestCommonAncestor(root->left, p, q);
    TreeNode* r = lowestCommonAncestor(root->right, p, q);
    if (l && r) return root;
    return (l ? l : r);
}


// https://leetcode.com/problems/merge-intervals/submissions/
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    vector<vector<int>> merged;
    sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) {return a[0] < b[0]; });

    int idx = 0;
    while (idx < intervals.size()) {
        if (!merged.empty() && merged.back()[1] >= intervals[idx][0]) {
            merged.back()[1] = max(intervals[idx][1], merged.back()[1]);
            idx++;
        }
        else {
            merged.push_back(intervals[idx]);
            idx++;
        }
    }
    return merged;
}


// https://leetcode.com/problems/sort-colors/
void sortColors(vector<int>& nums) {
    int pos = 0;

    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] == 0) {
            swap(nums[i], nums[pos]);
            pos++;
        }
    }
    for (int i = pos; i < nums.size(); i++) {
        if (nums[i] == 1) {
            swap(nums[i], nums[pos]);
            pos++;
        }
    }
}


// https://leetcode.com/problems/spiral-matrix/description/
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    int pos = 0, dir = 0, i = 0, j = 0;
    vector<int> ans;
    pair<int, int> dirs[4] = { {0, 1}, {1, 0}, {0, -1} , {-1, 0} };

    while (pos < m * n) {
        ans.push_back(matrix[i][j]);
        matrix[i][j] = INT_MAX;
        pos++;

        if (i + dirs[dir].first < 0 || i + dirs[dir].first >= m ||
            j + dirs[dir].second < 0 || j + dirs[dir].second >= n
            || matrix[i + dirs[dir].first][j + dirs[dir].second] == INT_MAX) {
            dir = (dir + 1) % 4;
        }
        i += dirs[dir].first; j += dirs[dir].second;
    }
    return ans;
}


// https://leetcode.com/problems/binary-tree-right-side-view/
vector<int> rightSideView(TreeNode* root) {
    vector<int> ans; 
    if (!root) return ans;

    queue<TreeNode*> Q;
    Q.push(root);

    while (!Q.empty()) {
        int num_nodes = Q.size();
        ans.push_back(Q.back()->val);

        for(int i = 0; i < num_nodes; i++){
            TreeNode* l = Q.front()->left;
            TreeNode* r = Q.front()->right;
            if (l) Q.push(l);
            if (r) Q.push(r);
            Q.pop();
        }
    }
    return ans;
}


// https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
TreeNode* treeBuilder(vector<int>& preorder, vector<int>& inorder, 
    unordered_map<int, int>& inorder_mp, int i, int pl, int pr) {
    int curr_node = preorder[i++];
    TreeNode* node = new TreeNode(curr_node);

    if (pl == pr) return node;

    int curr_node_pos = inorder_mp[curr_node];

    if (curr_node_pos != pl && curr_node_pos != pr) {
        node->left = treeBuilder(preorder, inorder, inorder_mp, i, pl, curr_node_pos - 1);
        node->right = treeBuilder(preorder, inorder, inorder_mp,
            i+ (curr_node_pos - pl), curr_node_pos + 1, pr);
    }
    else if (curr_node_pos == pl) {
        node->right = treeBuilder(preorder, inorder, inorder_mp, i, pl + 1, pr);
    }
    else {
        node->left = treeBuilder(preorder, inorder, inorder_mp, i, pl, pr - 1);
    }
    return node;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    unordered_map<int, int> inorder_mp;
    for (int i = 0; i < inorder.size(); i++) inorder_mp[inorder[i]] = i;
    return treeBuilder(preorder, inorder, inorder_mp, 0, 0, inorder.size() - 1);
}


// https://leetcode.com/problems/subsets/
void subsetsPermuter(vector<int>& nums, vector<vector<int>>& ans, vector<int>& path, int pos, int nums_size) { 
    ans.push_back(path);
    if (pos == nums.size()) return;

    for (int i = pos; i < nums_size; i++) {
        path.push_back(nums[i]);
        subsetsPermuter(nums, ans, path, i + 1, nums_size);
        path.pop_back();
    }
}
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> ans = {};
    int nums_size = nums.size();
    vector<int> path = {};
    subsetsPermuter(nums, ans, path, 0, nums_size);
    return ans;
}


// https://leetcode.com/problems/partition-equal-subset-sum/description/
bool canPartition(vector<int>& nums) {
    int sum = 0;
    for (auto n : nums) sum += n;
    
    int len = sum / 2;
    if (sum & 1) return false;
    vector<vector<bool>> dp(2, vector<bool>(len + 1));

    dp[1][0] = true, dp[0][0] = true;

    for (int i = 0; i < nums.size(); i++) {

        int n = nums[i];
        for (int j = 0; j <= len; j++) {
            if (j - n < 0) dp[i % 2][j] = dp[-(i % 2) + 1][j];
            else dp[i % 2][j] = (dp[-(i % 2) + 1][j - n] || dp[-(i % 2) + 1][j]);
        }

        if (dp[i % 2][len]) return true;
    }
    return false;
}


// https://leetcode.com/problems/design-add-and-search-words-data-structure/
class WordTrieNode {
    public:
        bool isWord;
        vector<WordTrieNode*> children;

        WordTrieNode() {
            isWord = false;
            children.resize(26, nullptr);
        }
};

bool wordTrieSearch(WordTrieNode* root, string& word, int pos) {
    if (pos == word.length()) return root->isWord;
    if (word[pos] == '.') {
        for (auto child : root->children) {
            if (wordTrieSearch(child, word, pos + 1)) return true;
        }
        return false;
    }
    if (root->children[word[pos] - 'a']) return wordTrieSearch(root->children[word[pos] - 'a'], word, pos + 1);
    return false;
}

class WordDictionary {
public:
    WordTrieNode* root;

    WordDictionary() {
        root = new WordTrieNode();
    }

    void addWord(string word) {
        WordTrieNode* tmp = root;
        for (char c : word) {
            if (!tmp->children[c - 'a']) tmp->children[c - 'a'] = new WordTrieNode();
            tmp = tmp->children[c - 'a'];
        }
        tmp->isWord = true;
    }

    bool search(string word) {
        return wordTrieSearch(root, word, 0);
    }
};



// https://leetcode.com/problems/word-break/
bool wordBreakHelper(string s, vector<string>& wordDict, vector<bool>& dp, int offset) {
    if (offset == s.size()) return true;
    if (dp[offset]) return false;
    dp[offset] = true;

    int slen = s.length() - offset;

    for (auto word : wordDict) {
        int wlen = word.length();
        if (wlen > slen) continue;

        int i = 0;
        for (; i < wlen; i++)
            if (s[i + offset] != word[i]) break;

        if (i == wlen && wordBreakHelper(s, wordDict, dp, offset + wlen)) return true;
    }
    return false;
}

bool wordBreak(string s, vector<string>& wordDict) {
    vector<bool> dp(s.size(), false);
    return wordBreakHelper(s, wordDict, dp, 0);
}


// https://leetcode.com/problems/insert-interval/submissions/
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
    vector<vector<int>> result;
    if (intervals.empty()) {
        result.push_back(newInterval);
        return result;
    }
    int start = newInterval[0], end = newInterval[1], n = intervals.size();
    int i = 0;
    while (i < n && start > intervals[i][1]) {
        result.push_back(intervals[i++]);
    }
    while (i < n && end >= intervals[i][0]) {
        start = min(start, intervals[i][0]);
        end = max(end, intervals[i][1]);
        i++;
    }
    result.push_back({ start,end });
    while (i < n) {
        result.push_back(intervals[i++]);
    }
    return result;
}

// https://leetcode.com/problems/reverse-linked-list/
ListNode* reverseList(ListNode* head) {
    if (!head) return nullptr;
    ListNode* prev = nullptr;
    ListNode* curr = head;

    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// https://leetcode.com/problems/linked-list-cycle/
bool hasCycle(ListNode* head) {
    unordered_set<ListNode*> table;
    while (head) {
        if (table.find(head) != table.end()) return true;
        table.insert(head);
        head = head->next;
    }
    return false;
}


//class TimeMap {
//private: 
//    unordered_map<string, vector<pair<int, string>>> mp;
//
//public:
//    TimeMap() {
//    }
//
//    void set(string key, string value, int timestamp) {
//        mp[key].emplace_back(timestamp, value);
//    }
//
//    string get(string key, int timestamp) {
//        auto it = upper_bound(mp[key].begin(), mp[key].end(), timestamp, [](pair<int, string> a) { return a.first; });
//        return it->first == timestamp ? it->second : prev(it)->second;
//    }
//};


// https://leetcode.com/problems/linked-list-cycle/
bool FloydsAlgorithm(ListNode* head) {
    if (!head) return false;
    ListNode* slow = head;
    ListNode* fast = head;

    while (slow && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}

// https://leetcode.com/problems/merge-two-sorted-lists/
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode* head = new ListNode(0);
    while (list1 && list2) {
        if (list1->val > list2->val) {
            head->next = list2;
            list2 = list2->next;
        }
        else {
            head->next = list1;
            list1 = list1->next;
        }
    }
    while (list1) {
        head->next = list1;
        list1 = list1->next;
    }
    while (list2) {
        head->next = list2;
        list2 = list2->next;
    }
    return head->next;
}

// https://leetcode.com/problems/clone-graph/
GraphNode* cloner(GraphNode* curr_node, unordered_map<int, GraphNode*> &visited) {
    if (visited.find(curr_node->val) != visited.end()) return visited[curr_node->val];
    cout << curr_node->val << endl;

    GraphNode* new_node = new GraphNode(curr_node->val);
    visited[curr_node->val] = new_node;
    
    new_node->neighbors.reserve(curr_node->neighbors.size());
    for (auto n : curr_node->neighbors) {
        new_node->neighbors.push_back(cloner(n, visited));
    }
    return new_node;
}

GraphNode* cloneGraph(GraphNode* node) {
    if (!node) return node;
    unordered_map<int, GraphNode*> visited;
    return cloner(node, visited);
}

// https://leetcode.com/problems/evaluate-reverse-polish-notation/
int evalRPN(vector<string>& tokens) {
    stack<int> nums;
    for (int i = 0; i < tokens.size(); i++) {
        if (isdigit(tokens[i][0]) || (tokens[i].length() > 1 && tokens[i][0] == '-')) {
            nums.push(stoi(tokens[i]));
        }
        else {
            int num1 = nums.top();
            nums.pop();
            int num2 = nums.top();
            nums.pop();
            int result = 0;

            if (tokens[i] == "+") nums.push(num2 + num1);
            else if (tokens[i] == "-") nums.push(num2 - num1);
            else if (tokens[i] == "*") nums.push(num2 * num1);
            else nums.push(num2 / num1);
        }
    }
    return nums.top();
}

// https://leetcode.com/problems/longest-repeating-character-replacement/
//int characterReplacement(string s, int k) {
//    int ptr1 = 0, ptr2 = 0, maxf = 0, maxLen = 0;
//    unordered_map<char, int> table;
//    while (ptr2 < s.length()) {
//        int sublen = ptr2 - ptr1 + 1;
//        // increment the occurence of the new character in our table
//        table[s[ptr2]]++;
//
//        // maxf keeps track of the running highest frequency of any character in current and past substrings
//        maxf = max(maxf, table[s[ptr2]]);
//        
//        // if current substring length - max frequency of a character in substring > k
//        // that means we need to bring the sliding window closer from the back
//        if (sublen - maxf > k) {
//            table[s[ptr1]]--;
//            ptr1++; sublen--;
//        }
//        maxLen = max(maxLen, sublen); ptr2++;
//    }
//    return maxLen;
//}


int characterReplacement(string s, int k) {
    int n = s.length(), l = 0, r = 0;
    int max_len = 0;
    int most_common = 0;
    vector<int> mp(26, 0);

    while (r < n) {

        mp[s[r] - 'A']++;
        if (mp[s[r] - 'A'] > most_common) {
            most_common = mp[s[r] - 'A'];
        }
        else if (--k < 0) {
            mp[s[l++] - 'A']--;
            k++;
        }
        max_len = max(max_len, r - l + 1);
        r++;
    }

    return max_len;
}


// https://leetcode.com/problems/minimum-window-substring/
string minWindow(string s, string t) {
    int start = 0, end = 0, counter = t.length(), minStart = 0, minLen = INT32_MAX;
        vector<int> m(128, 0);
        for (char c : t) m[c]++;

        while (end < s.length()) {
            if (m[s[end]] > 0) counter--;
            m[s[end]]--;

            while (counter == 0) {
                if(end-start + 1 < minLen){
                    minLen = end-start + 1;
                    minStart = start;
                }
                m[s[start]]++;
                if (m[s[start]] > 0) counter++;
                start++;
            }
            end++;
        }
        return minLen == INT32_MAX ? "" : s.substr(minStart, minLen);
}





// https://leetcode.com/problems/word-break/submissions/
//bool wordBreak(string s, vector<string>& wordDict) {
//    unordered_map<string, bool> memo;
//    return breakHelper(s, wordDict, memo);
//}
//bool breakHelper(string s, vector<string>& wordDict, unordered_map<string, bool>& memo) {
//    if (s.empty()) return true;
//    if (memo.find(s) != memo.end()) return memo[s];
//
//    for (string& word : wordDict) {
//        if (s.length() >= word.length() && s.substr(0, word.length()) == word) {
//            if (breakHelper(s.substr(word.length()), wordDict, memo)) {
//                memo[s] = true;
//                return true;
//            }
//        }
//    }
//    memo[s] = false;
//    return false;
//}


// https://leetcode.com/problems/combination-sum/submissions/
void combinationDFS(vector<int>& candidates, vector<vector<int>>& ans, vector<int>& path, int target, int pos, int len) {
    if (target == 0) ans.push_back(path);
    if (target < 0) return;

    for (int i = pos; i < len; i++){
        path.push_back(candidates[i]);
        combinationDFS(candidates, ans, path, target - candidates[i], i, len);
        path.pop_back();
    }
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> ans;
    vector<int> path;
    combinationDFS(candidates, ans, path, target, 0, candidates.size());
    return ans;
}


// https://leetcode.com/problems/combination-sum-iv/
int combinationSum4(vector<int>& nums, int target) {
    unordered_map<int, int> table;
    return combinationHelper(nums, target, table);
}

int combinationHelper(vector<int>& nums, int target, unordered_map<int, int> &table) {
    if (target == 0) return 1;
    if (table.find(target) != table.end()) return table[target];

    int totalWays = 0;
    for (int i : nums) {
        if (target - i >= 0) {
            totalWays += combinationHelper(nums, target - i, table);
        }
    }
    table[target] = totalWays;
    return totalWays;
}

// https://leetcode.com/problems/k-closest-points-to-origin/
class CoordCompare
{
public:
    bool operator() (vector<int> p1, vector<int> p2)
    {
        return (pow(p1[0], 2) + pow(p1[1], 2)) < (pow(p2[0], 2) + pow(p2[1], 2));
    }
};

vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
    priority_queue<vector<int>, vector<vector<int>>, CoordCompare> closest;
    for (auto p : points) {
        closest.push(p);
        if (closest.size() > k) closest.pop();
    }

    vector<vector<int>> ans(closest.size());
    for (int i = 0; i < k; i++) {
        ans[i] = closest.top();
        closest.pop();
    }

    return ans;
}


// https://leetcode.com/problems/01-matrix/
vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
    int rows = mat.size(), cols = mat[0].size();
    vector<vector<int>> dists(rows, vector<int>(cols, INT_MAX));
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (mat[r][c] == 0) dists[r][c] = 0;
            else {
                if (r - 1 >= 0) dists[r][c] = min(1 + dists[r - 1][c], dists[r][c]);
                if (c - 1 >= 0) dists[r][c] = min(1 + dists[r][c - 1], dists[r][c]);
            }
        }
    }
    for (int r = rows - 1; r >= 0; r--) {
        for (int c = cols - 1; c >= 0; c--) {
            if (r + 1 < rows) dists[r][c] = min(1 + dists[r + 1][c], dists[r][c]);
            if (c + 1 < cols) dists[r][c] = min(1 + dists[r][c + 1], dists[r][c]);
        }
    }
    return dists;
}


// https://leetcode.com/problems/group-anagrams/description/
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, int> words_to_idx;
    vector<vector<string>> ans;
    int pos = 0;
    for (auto s : strs) {
        string tmp = s;
        sort(s.begin(), s.end());
        if (words_to_idx.find(s) != words_to_idx.end()) ans[words_to_idx[s]].push_back(tmp);
        else {
            ans.push_back({ tmp });
            words_to_idx[s] = pos++;
        }
    }
    return ans;
}

string countingSort(string s) {
    string sorted = "";
    vector<int> alphabet(26, 0);
    for (char c : s) alphabet[c - 'a']++;
    for (int i = 0; i < 26; i++) {
        while (alphabet[i]-- > 0) {
            sorted += char('a' + i);
        }
    }
    return sorted;
}
vector<vector<string>> groupAnagrams2(vector<string>& strs) {
    unordered_map<string, vector<string>> anagrams;
    for (string s : strs) {
        anagrams[countingSort(s)].push_back(s);
    }
    
    vector<vector<string>> ans;
    for (auto strs : anagrams)
        ans.push_back(strs.second);
    
    return ans;
}

//int lengthOfLongestSubstring(string s) {
//
//    int n = s.length();
//    int max_len = 0;
//    map<char, int> temp;
//
//    for (int i = 0, j = 0; j < n; j++) {
//        if (temp.find(s[j]) != temp.end()) {
//            i = max(i, temp[s[j]] + 1);
//            temp[s[j]] = j;
//        }
//        max_len = max(max_len, j - i + 1);
//        temp.insert({ s[j], j });
//    }
//    return max_len;
//}

// https://leetcode.com/problems/longest-substring-without-repeating-characters/
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> mp;
    int len = 0, longest = 0;
    for (int i = 0; i < s.length(); i++) {
        if (mp.find(s[i]) == mp.end()) {
            mp[s[i]] = i;
            len++;
        }
        else if (mp[s[i]] < i - len) {
            mp[s[i]] = i;
            len++;
        }
        else {
            len = i - mp[s[i]];
            mp[s[i]] = i;
        }
        longest = max(len, longest);
    }
    return longest;
}

// https://leetcode.com/problems/3sum/
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    int i = 0;
    while (i < nums.size() - 2 && nums[i] <= 0) {
        int l = i + 1, r = nums.size() - 1;
        int num1 = nums[i++];
        while (l < r) {
            int tot = num1 + nums[l] + nums[r];
            if (tot < 0) l++;
            else if (tot > 0) r--;
            else {
                ans.push_back({ num1, nums[l], nums[r] });
                l++; r--;
                while (l < nums.size() && nums[l] == nums[l - 1]) l++;
                while (r >= 0 && nums[r] == nums[r + 1]) r--;
            } 
        }
        while (i < nums.size() - 2 && nums[i] == nums[i-1]) i++;
    }
    return ans;
}


// https://leetcode.com/problems/squares-of-a-sorted-array/
vector<int> sortedSquares(vector<int>& nums) {
    vector<int> sortedSqrs(nums.size(), 0);
    int l = 0, r = nums.size() - 1;
    while (l <= r) {
        int sq1 = pow(nums[l], 2), sq2 = pow(nums[r], 2);
        if (sq1 > sq2) {
            sortedSqrs[r - l] = sq1;
            l++; 
        }
        else {
            sortedSqrs[r - l] = sq2;
            r--;
        }
    }
    return sortedSqrs;
}


// https://leetcode.com/problems/product-of-array-except-self/
vector<int> productExceptSelf(vector<int>& nums) {
    int l = 1, r = 1;
    size_t len = nums.size();
    vector<int> products(len, 1);
    for (size_t i = 0; i < len; i++) {
        cout << l << " " << r << endl;
        products[i] *= l;
        l *= nums[i];
        products[len - i - 1] *= r;
        r *= nums[len - i - 1];
    }
    return products;
}


// https://leetcode.com/problems/number-of-islands/
void islandBFS(vector<vector<char>>& grid, vector<pair<int, int>>& directions, int r, int c, int rows, int cols) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (grid[r][c] == '_' || grid[r][c] == '0') return;
    grid[r][c] = '_';
    for (auto p : directions) islandBFS(grid, directions, r + p.first, c + p.second, rows, cols);
}


int numIslands(vector<vector<char>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    vector<pair<int, int>> directions = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
    int islands = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == '_') continue;
            if (grid[i][j] == '1') {
                islandBFS(grid, directions, i, j, rows, cols);
                islands++;
            }
            grid[i][j] = '_';
        }
    }

    return islands;
}


// https://leetcode.com/problems/path-sum-ii/
void pathSumDFS(TreeNode* root, int targetSum, vector<int>& path, vector<vector<int>>& ans) {
    if (!root->left && !root->right) {
        if (targetSum == 0) ans.push_back(path);
        return;
    }
    if (root->left) {
        int l = root->left->val;
        path.push_back(l);
        pathSumDFS(root->left, targetSum - l, path, ans);
        path.pop_back();
    }
    if (root->right) {
        int r = root->right->val;
        path.push_back(r);
        pathSumDFS(root->right, targetSum - r, path, ans);
        path.pop_back();
    }
}
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    if (!root) return { };
    vector<vector<int>> ans;
    vector<int> path = {root->val};
    pathSumDFS(root, targetSum - root->val, path, ans);
    return ans;
}

// https://leetcode.com/problems/swap-nodes-in-pairs/
ListNode* swapPairs(ListNode* head) {
    if (!head) return nullptr;
    if (!head->next) return head;

    ListNode* prev = nullptr;
    ListNode* first = head;
    ListNode* second = head->next;
    head = second;
    while (second) {
        ListNode* third = second->next;
        second->next = first;
        first->next = third;
        if (prev) prev->next = second;

        prev = first;
        first = third;
        second = (third ? third->next : nullptr);
    }
    return head;
}


// https://leetcode.com/problems/rotate-array/description/
void rotate(vector<int>& nums, int k) {
    k = k % nums.size();
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.end());
}


// https://leetcode.com/problems/contiguous-array/
int findMaxLength(vector<int>& nums) {
    int n = nums.size();
    vector<int> mp(n * 2 + 1, -2);
    mp[n] = -1;
    int balance = 0, longest = 0;

    for (int i = 0; i < nums.size(); i++) {
        balance += (nums[i] ? 1 : -1);
        if (mp[balance + n] < -1) mp[balance + n] = i;
        else longest = max(longest, i - mp[balance + n]);
    }
    return longest;
}


// https://leetcode.com/problems/maximum-width-of-binary-tree/
int widthOfBinaryTree(TreeNode* root) {
    queue<pair<long, TreeNode*>> levels;
    levels.emplace(0, root);
    
    int max_width = 1;
    while (!levels.empty()) {
        int n = levels.size();
        int curr_start = levels.front().first;
        max_width = max(max_width, int(levels.back().first - curr_start + 1));

        curr_start -= curr_start % 2;
        for (int i = 0; i < n; i++) {
            long pos = levels.front().first;
            TreeNode* curr = levels.front().second;
            levels.pop();

            if (curr->left) levels.emplace(pos * 2 - curr_start, curr->left);
            if (curr->right) levels.emplace(pos * 2 + 1 - curr_start, curr->right);
        }
    }
    return max_width;
}


// https://leetcode.com/problems/decode-string/
string decodeString(string s) {
    stack<int> multiplier;
    stack<string> strings;

    int pos = 0;
    string tmp = "", tmp_str = "";
    while (pos < s.length()) {

        if (!strings.empty()) cout << strings.top() << endl;

        if (isdigit(s[pos])) {
            tmp += s[pos];
        }
        else if (s[pos] == '[') {
            multiplier.push(stoi(tmp));
            strings.push(tmp_str);
            tmp = tmp_str = "";
        }
        else if (s[pos] == ']') {
            int n = multiplier.top();
            string curr = tmp_str;
            for (int i = 1; i < n; i++) tmp_str += curr;

            tmp_str = strings.top() + tmp_str;
            multiplier.pop();
            strings.pop();
        }
        else {
            tmp_str += s[pos];
        }
        pos++;
    }

    return tmp_str;
}


// https://leetcode.com/problems/odd-even-linked-list/
ListNode* oddEvenList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* odd = head;
    ListNode* even = head->next;
    ListNode* even_head = even;
    ListNode* curr = even->next;
    int pos = 1;
    
    while (curr) {
        cout << odd->val << " - " << even->val << endl;

        if (pos++ % 2) {
            odd->next = curr;
            odd = odd->next;
        }
        else {
            even->next = curr;
            even = even->next;
        }
        curr = curr->next;
    }
    even->next = nullptr;
    odd->next = even_head;
    return head;
}


// https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
int findMin(vector<int>& nums) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);
    int left = 0, mid = nums.size() / 2, right = nums.size() - 1;
    while (right - left > 1) {
        nums[mid] < nums[right] ? right = mid : left = mid;
        mid = (left + right) / 2;
    }
    return min(nums[left], nums[right]);
}


// https://leetcode.com/problems/search-in-rotated-sorted-array/
int searchRotatedSortedArray(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;

    while (l < r) {
        int m = (l + r) / 2;
        int left = nums[l], right = nums[r], mid = nums[m];

        if (target == mid) return m;

        if (left < right) {
            if (target < mid) r = m - 1;
            else l = m + 1;
        }
        else if (left <= mid) {
            if (target < mid && target >= left) r = m - 1;
            else l = m + 1;
        }
        else {
            if (target <= right && target > mid) l = m + 1;
            else r = m - 1;
        }
    }
    return nums[l] == target ? l : -1;
}


int orangesRotting(vector<vector<int>>& grid) {
    vector<pair<int, int>> directions = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
    queue< pair<int, int>> rotten;
    int rows = grid.size(), cols = grid[0].size();
    int minutes = 0, fresh = 0;
    
    for (auto i = 0; i < rows; i++) {
        for (auto j = 0; j < cols; j++) {
            if (grid[i][j] == 2) rotten.emplace(i, j); 
            if (grid[i][j] == 1) fresh++;
        }
    }

    while (fresh != 0 && !rotten.empty()) {
        queue< pair<int, int>> rotten2;
        while (!rotten.empty()) {
            int r = rotten.front().first, c = rotten.front().second;
            rotten.pop();
            for (auto p : directions) {
                if (r + p.first >= 0 && r + p.first < rows && 
                    c + p.second >= 0 && c + p.second < cols &&
                    grid[r + p.first][c + p.second] == 1) {
                    fresh--;
                    grid[r + p.first][c + p.second] = 2;
                    rotten2.emplace(r + p.first, c + p.second);
                }
            }
        }
        rotten = rotten2;
        minutes++;
    }
    return fresh == 0 ? minutes : -1;
}


// https://leetcode.com/problems/validate-binary-search-tree/
void inOrderTraversal(TreeNode* root, vector<int>& result) {
    if (root->left) inOrderTraversal(root->left, result);
    result.push_back(root->val);
    if (root->right) inOrderTraversal(root->right, result);
}
bool isValidBST(TreeNode* root) {
    vector<int> result;
    inOrderTraversal(root, result);
    
    for (size_t i = 1; i < result.size(); i++) {
        if (result[i - 1] >= result[i]) return false;
    }
    return true;
}


// https://leetcode.com/problems/valid-parentheses/
bool isValid(string s) {
    ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int b1 = 0, b2 = 0, b3 = 0;
    stack<char> prevChar;

    for (char c : s) {
        switch (c) {
        case '(':
            prevChar.push(c);
            break;
        case ')':
            if (prevChar.top() != '(') return false;
            prevChar.pop();
            break;
        case '[':
            prevChar.push(c);
            break;
        case ']':
            if (prevChar.top() != '[') return false;
            prevChar.pop();
            break;
        case '{':
            prevChar.push(c);
            break;
        case '}':
            if (prevChar.top() != '{') return false;
            prevChar.pop();
            break;
        default: break;
        }
    }
    return true;
}

// https://leetcode.com/problems/reverse-bits/
uint32_t reverseBits(uint32_t n) {
    uint32_t reversed = 0;
    for (int i = 31; i >= 0; i--) {
        reversed <<= 1;
        reversed |= (n & 1);
        n >>= 1;
    }
    return reversed;
}

// https://leetcode.com/problems/subtree-of-another-tree/
bool compareTree(TreeNode* root, TreeNode* subRoot) {
    if (!root && !subRoot) {
        return true;
    }
    else if (!subRoot || !root) {
        return false;
    }
    else if (root->val == subRoot->val) {
        return compareTree(root->left, subRoot->left) && compareTree(root->right, subRoot->right);
    }
    return false;
}
bool isSubtree(TreeNode* root, TreeNode* subRoot) {
    cout << "root: " << root->val << "  subRoot: " << subRoot->val << endl;
    if (!root && !subRoot) {
        return true;
    }
    else if (!subRoot || !root) {
        return false;
    }
    else if (root->val == subRoot->val) {
        return compareTree(root, subRoot) || isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
    }
    return isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
}

// https://leetcode.com/problems/palindromic-substrings/
int countSubstrings(string s) {
    int count = 0;
    for (int i = 0; i < s.length(); i++) {
        count++;

        // check for even-length palindromes
        int ptr1 = i, ptr2 = i+1;
        while(ptr2 < s.length() && ptr1 >= 0 && s[ptr1--] == s[ptr2++]) count++;

        // check for odd-length palindromes
        ptr1 = i - 1, ptr2 = i + 1;
        while (ptr2 < s.length() && ptr1 >= 0 && s[ptr1--] == s[ptr2++]) count++;
    }
    return count;
}

// https://leetcode.com/problems/invert-binary-tree/
TreeNode* invertTree(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        TreeNode* parent = q.front();
        q.pop();
        if (parent) {
            TreeNode* tmp = parent->left;
            q.push(tmp); q.push(parent->right);
            parent->left = parent->right;
            parent->right = tmp;
        }
    }
    return root;
}

// https://leetcode.com/problems/same-tree/
bool isSameTree(TreeNode* p, TreeNode* q) {
    queue<TreeNode*> q1, q2;
    q1.push(p); q2.push(q);

    while (!q1.empty() && !q2.empty()) {
        TreeNode* p1 = q1.front();
        q1.pop();
        TreeNode* p2 = q2.front();
        q2.pop();
        if (p1 && !p2 || !p1 && p2) return false;
        if (p1 && p2 && (p1->val != p2->val)) return false;

        if (p1) {
            q1.push(p1->left); q1.push(p1->right);
            q2.push(p2->left); q2.push(p2->right);
        }
    }
    return q2.empty() && q1.empty();
}


int minDifference(vector<int> v1, vector<int> v2) {
    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());

    int ptr1 = 0, ptr2 = 0, minDiff = INT32_MAX;

    while (ptr1 != v1.size() && ptr2 != v2.size()) {
        minDiff = min(minDiff, abs(v1[ptr1] - v2[ptr2]));
        
        if (v1[ptr1] > v2[ptr2]) {
            ptr2++;
        }
        else if (v1[ptr1] < v2[ptr2]) {
            ptr1++;
        }
        else return 0;
    }
    return minDiff;
}
/*
TreeNode * randNode(TreeNode* root) {
    vector<TreeNode*> nodes;
    queue<TreeNode*> BFS;
    BFS.push(root);

    while (!BFS.empty()) {
        TreeNode* top = BFS.front();
        BFS.pop();
        if (top) {
            nodes.push_back(top);
            BFS.push(top->left);
            BFS.push(top->right);
        }
    }
    
    int r = rand();
    return nodes[r % nodes.size()];
}

TreeNode* randNode2(TreeNode* root) {
    size_t size = root->size;
    int target = rand() % size;
    TreeNode* currNode = root;
    while (true) {
        int leftSize = 0;
        if (currNode->left) leftSize = root->left->size;
        
        if (target == 0) return currNode;
        else if (target > leftSize) {
            currNode = currNode->right;
        }
        else {
            currNode = currNode->left;
        }
    }
    return nullptr;
}
*/
// find max subarray
// [-5,3,4,-2]


int maxSubarray(vector<int> &v) {
    int maxSum = INT32_MIN, currSum = 0;
    for (int i : v) {
        currSum += i;
        maxSum = max(maxSum, currSum);
        if (currSum < 0) currSum = 0;
    }
    return maxSum;
}

int maxSubarrayProduct(vector<int>& v) {
    int maxP = 1, minP = 1, currP = 1;
    int totalMaxP = INT32_MIN;
    for (int i : v) {
        if (i < 0) {
            swap(minP, maxP);
        }
        maxP = max(maxP * i, i);
        totalMaxP = max(maxP, totalMaxP);
        minP = min(minP * i, i);
    }
    return totalMaxP;
}





// https://leetcode.com/problems/task-scheduler/
int leastInterval(vector<char>& tasks, int n) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);
    unordered_map<char, int> m;
    int maxCount = 0;
    for (char c : tasks) {
        m[c]++;
        maxCount = max(maxCount, m[c]);
    }
    int ans = (maxCount - 1) * (n + 1);
    for (pair<char, int> p : m)
        if (p.second == maxCount) ans++;
    return max(int(tasks.size()), ans);
}























//################################################################################################
//################################################################################################
//###################################                       ######################################
//###################################   EECS 281 Practice   ######################################
//###################################                       ######################################
//################################################################################################
//################################################################################################

// https://umich.instructure.com/courses/491173/files/folder/Exam%20-%20Practice/Exam%202%20-%20Final?preview=24280664
int number_of_tilings(int n) {
    // base cases f(1) = 1, f(2) = 3
    if (n == 1) return 1;
    if (n == 2) return 3;

    // f(n) = f(n-1) + 2*f(n-2)
    int base1 = 1, base2 = 3, curr = 3;
    for (int i = 3; i <= n; i++) {
        curr = base2 + 2 * base1;
        base1 = base2;
        base2 = curr;
    }
    return curr;
}

// https://umich.instructure.com/courses/491173/files/folder/Exam%20-%20Practice/Exam%202%20-%20Final?preview=24280664
bool existsInTree(TreeNode* root, int val) {
    if (!root) return false;
    if (root->val == val) return true;
    return root->val > val ? existsInTree(root->left, val) : existsInTree(root->right, val);
}

// file:///C:/Users/brian/Downloads/practice_final_2_combined_answers.pdf
bool zero_contiguous_sum(vector<int>& nums) {
    unordered_set<int> table;
    table.insert(0);
    int currSum = 0;
    for (int i : nums) {
        currSum += i;
        if (table.find(currSum) != table.end()) return true;
        table.insert(currSum);
    }
    return false;
}

// file:///C:/Users/brian/Downloads/practice_final_2_combined_answers.pdf
void range_queries(const vector<unsigned int>& data,
    const vector<Query>& queries) {
    
    unordered_map<int, vector<int>> table;
    vector<int> v(data.size(), 0);
    for (int i = 0; i < data.size(); i++) {
        int num = data[i];
        int count = 0;
        if (table.find(num) == table.end()) {
            table[num].resize(i+1, 0);
            for (int j = i; j < data.size(); j++) {
                if (data[j] == num) 
                    count++;
                table[num].push_back(count);
            }
        }
    }

    for (const Query& q : queries) {
        if (table[q.id].empty()) cout << "0 ";
        else cout << table[q.id][q.end+1] - table[q.id][q.start] << " ";
    }
}


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
bool cmp_Intervals(const Interval& a, const Interval& b) { return (a.start < b.start); }

vector<Interval> merge_intervals1(vector<Interval>& vec) {
    std::sort(vec.begin(), vec.end(), cmp_Intervals);

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


// https://leetcode.com/problems/course-schedule/
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> G(numCourses, vector<int>());
    vector<int> edges(numCourses, 0);
    for ( auto v : prerequisites) {
        G[v[1]].push_back(v[0]);
        edges[v[0]]++;
    }

    vector<int> clean;
    for (int i = 0; i < edges.size(); i++)
        if (!edges[i]) clean.push_back(i);

    for (int i = 0; i < clean.size(); i++) {
        for (auto j : G[clean[i]]) {
            --edges[j];
            if (edges[j] == 0) {
                clean.push_back(j);
            }
        }
    }
    return clean.size() == numCourses;
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


// https://leetcode.com/problems/coin-change/
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    for (int i = 1; i < amount + 1; i++) {
        for (auto c : coins) {
            if (i - c >= 0 && dp[i - c] != INT_MAX)
                dp[i] = min(dp[i], dp[i - c] + 1);
        }
    }
    return (dp[amount] == INT_MAX ? -1 : dp[amount]);
}


class TrieNode {
public:
    TrieNode* children[26];
    bool period;
    TrieNode() {
        period = false;
        for (int i = 0; i < 26; i++) {
            children[i] = NULL;
        }
    }
};

class Trie {
public:
    TrieNode* root;
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        TrieNode* node = root;
        for (auto c : word) {
            int pos = c - 'a';
            if (node->children[pos] == NULL) node->children[pos] = new TrieNode();
            node = node->children[pos];
        }
        node->period = true;
    }

    bool search(string word) {
        TrieNode* node = root;
        for (auto c : word) {
            int pos = c - 'a';
            if (node->children[pos] == NULL) return false;
            node = node->children[pos];
        }
        return node->period;
    }

    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (auto c : prefix) {
            int pos = c - 'a';
            if (node->children[pos] == NULL) return false;
            node = node->children[pos];
        }
        return true;
    }
};



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

// You are given k sorted linked lists. Write a program that merges all k lists into a single sorted list.
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

// You are given a vector of integers, vec, and you are told to implement a function that moves all elements
// with a value of 0 to the end of the vector while maintaining the relative order of the non - zero elements.
void shift_zeros(vector<int>& vec) {

    if (vec.size() > 1) {
        for (int ptr1 = 0, ptr2 = 1; ptr2 < vec.size(); ptr1++, ptr2++) {
            if (vec.at(ptr1) == 0) {
                while (vec.at(ptr2) == 0 && ptr2 < vec.size() - 1) {
                    ptr2++;
                }
                if (vec.at(ptr2) != 0) {
                    swap(vec.at(ptr1), vec.at(ptr2));
                }
                else {
                    break;
                }
            }
        }
    }
}

// You are given a vector of integers, temps, that stores the daily temperature forecasts for the next few
// days.Write a program that, for each index of the input vector, stores the number of days you need to wait
// for a warmer temperature.If there is no future day where this is possible, a value of 0 should be stored.

vector<int> warmer_temperatures(vector<int>& temps) {
    vector<int> ans(temps.size());

    for (int ptr1 = temps.size() - 1; ptr1 >= 0; ptr1--) {
        int ptr2 = ptr1 + 1;
        while (ptr2 < temps.size() && temps[ptr1] >= temps[ptr2]) {
            if (ans[ptr2] > 0) {
                ptr2 += ans[ptr2];
            }
            else
                ptr2 = temps.size();
        }
        if (ptr2 != temps.size()) {
            ans.at(ptr1) = ptr2 - ptr1;
        }
    }
    return ans;
}

// You are given a m x n matrix in the form of a vector of vectors that has the following properties:
// • integers in each row are sorted in ascending order from left to right
// • integers in each column are sorted in ascending order from top to bottom
// Write a function that searches for a value in this matrixand returns whether the element can be found.
bool matrix_search(vector<vector<int>>& matrix, int target) {
    int mid_col = matrix.at(0).size() / 2;
    int row = 0;
    // linear search on middle column so we can isolate two of the four quadrants for our search
    for (row = 0; row < matrix.size(); ++row) {
        if (matrix.at(row).at(mid_col) > target)
            break;
        // we accidentally found target
        else if (matrix.at(row).at(mid_col) == target)
            return true;
    }

    // target can only be in last row and after mid_col
    if (row > matrix.size()) row--;
    // found the row to use to isolate the two quadrants

    for (int i = 0; i < row; i++)
        for (int j = mid_col + 1; j < matrix.at(0).size(); j++)
            if (matrix.at(i).at(j) == target)
                return true;
    for (int i = row; i < matrix.size(); i++)
        for (int j = 0; j < mid_col; j++)
            if (matrix.at(i).at(j) == target)
                return true;
    return false;
}



// Prints out all different pairs in input_vec that have same sum.
// Time Complexity: O(n^2) on average
void two_pair_sums(const vector<int>& nums, ostream& os) {

    unordered_map<int, pair<int, int>> hash;

    for (int i = 0; i < nums.size(); i++) {
        for (int j = i + 1; j < nums.size(); j++) {

            int sum = nums.at(i) + nums.at(j);

            if (hash.find(sum) != hash.end()) {
                os << "(" << hash.at(sum).first << ", " << hash.at(sum).second << ")";
                os << " and " << "(" << nums.at(i) << ", " << nums.at(j) << ")\n";
            }

            hash[sum] = pair<int, int>{ nums.at(i),nums.at(j) };
        }
    }
}


// EECS 281 Lab7 Written Problem
// Given a dictionary consisting of many prefixes and a sentence, you need to replace all the successors in
// the sentence with the prefix forming it.If a successor has many prefixes that can form it, replace it with
// the prefix with the shortest length.
// https://umich.instructure.com/courses/491173/files/folder/Lab/lab07/Replace%20Words%20Written%20Problem?preview=23559563
// Time Complexity: O(PM+NM^2), where 
// P = number of prefixes in prefix vector 
// N = number of words in sentence vector
// M = the length of the longest string in the sentence vector

vector<string> replace_words(const vector<string>& prefixes,
    const vector<string>& sentence) {

    unordered_set<string> prefix_list;
    vector<string> ans;

    for (string p : prefixes) {
        prefix_list.insert(p);
    }

    for (int j = 0; j < sentence.size(); j++) {
        bool found = false;
        for (int i = 1; i <= sentence.at(j).length(); i++) {
            auto iter = prefix_list.find(sentence.at(j).substr(0, i));
            if (iter != prefix_list.end()) {
                ans.push_back(sentence.at(j).substr(0, i));
                found = true;
                break;
            }
        }
        if (!found) {
            ans.push_back(sentence.at(j));
        }
    }

    return ans;
}


class MinStack {
public:
    vector<int> S;
    int min_val = INT_MAX;

    MinStack() {
    }

    void push(int val) {
        S.push_back(val);
        min_val = min(val, min_val);
    }

    void pop() {
        if (min_val == S.back()) {
            min_val = INT_MAX;
            S.pop_back();
            for (size_t i = 0; i < S.size(); i++) {
                min_val = min(min_val, S[i]);
            }
        }
        else {
            S.pop_back(); 
        }
    }

    int top() {
        return S.back();
    }

    int getMin() {
        return min_val;
    }
};