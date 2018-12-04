#include "shader_module.h"
#include "filesystem.h"
#include "macro.h"

ShaderModule CreateShaderModule(VkDevice device, const char* filepath) {
    auto code = FileRead(filepath);
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateShaderModule(device, &info, nullptr, &module));

    SpirvReflections reflections = ReflectShader(code.data(), code.size());
    
    return ShaderModule {
        module,
        std::move(reflections)
    };
}
