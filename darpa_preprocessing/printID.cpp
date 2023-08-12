#include <iostream>
#include <string>
#include <xxhash.h>

uint64_t hashString(const std::string& input) {
    uint64_t seed = 0xABCD;

    // Calculate the hash using xxHash
    uint64_t hash = XXH64(input.c_str(), input.length(), seed);

    return hash;
}

int main(int argc, char *argv[]) {
    std::string input = (argv[1]);
    uint64_t hashValue = hashString(input);

    std::cout << "Input: " << input << std::endl;
    std::cout << "Hash: " << hashValue << std::endl;

    return 0;
}

