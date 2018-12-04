#include "filesystem.h"

#include <fstream>
#include "macro.h"

std::vector<char> FileRead(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    ASSERT(file.is_open());

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}
