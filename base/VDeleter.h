#pragma once

#include <functional>
#include <vulkan/vulkan.h>

// deprecated note that https://github.com/Overv/VulkanTutorial/issues/32

template <typename T>
class VDeleter
{
public:

    VDeleter() : VDeleter([](T, VkAllocationCallback*) {}) 
    {
    }

    VDeleter(std::function<void(T, VkAllocationCallbacks*)> deletef)
    {
        this->deleter = [=](T obj) { deletef(obj, nullptr); };
    }

    virtual ~VDeleter()
    {
        cleanup();
    }

    const T* operator &() const
    {
        return &object;
    }

    T* replace()
    {
        cleanup();
        return &object;
    }

    operator T() const
    {
        return object;
    }

private:

    T object { VK_NULL_HANDLE };
    std::function<void(T)> deleter;

    void cleanup()
    {
        if (object != VK_NULL_HANDLE)
            deleter(object);
        object = VK_NULL_HANDLE;
    }
};
