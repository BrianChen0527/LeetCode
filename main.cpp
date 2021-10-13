using namespace std;
#include "Functions.h"



int main(int argc, char* argv[]){
    
    std::map<char, int> romanNumerals = {
        {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000},
    };
    /*
    std::map<int, char> romanNumerals = {
        {1, 'I'},
        {5, 'V'},
        {10, 'X'},
        {50, 'L'},
        {100, 'C'},
        {500, 'D'},
        {1000, 'M'}
    };
    */
    auto item = romanNumerals.find('I');
    cout << item->second;

}

