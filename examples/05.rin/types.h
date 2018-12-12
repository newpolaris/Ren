#pragma once

#include <initializer_list>
#include <vector>
#include <volk.h>

struct BufferCreateInfo;
struct Buffer;

struct ShaderModule;
using ShaderModules = std::initializer_list<ShaderModule>;

using ImageViewList = std::vector<VkImageView>;
using QueueFamilyProperties = std::vector<VkQueueFamilyProperties>;
