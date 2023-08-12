#include "hasher.h"

uint64_t hashString(const std::string &input)
{
    uint64_t seed = 0xABCD;

    // Calculate the hash using xxHash
    uint64_t hash = XXH64(input.c_str(), input.length(), seed);

    return hash;
}