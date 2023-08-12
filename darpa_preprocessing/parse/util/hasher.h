#ifndef HASHER_H_
#define HASHER_H_

#include <string>
#include <cinttypes>
#include <xxhash.h>

uint64_t hashString(const std::string &input);

#endif