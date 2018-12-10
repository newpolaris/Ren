#pragma once

#include <initializer_list>
#include <vector>

struct BufferCreateInfo;
struct Buffer;

struct ShaderModule;
using ShaderModules = std::initializer_list<ShaderModule>;

using ImageViewList = std::vector<VkImageView>;
