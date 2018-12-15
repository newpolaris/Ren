# Rin

Test vulkan renderer.
The goal is to experiment with a rendering techniques, such as cone culling, spirv reflection, task/mesh shading which mentioned in youtube stream 'niagara'

# Issues

* NVidia GT 750m GPU cause VK_ERROR_DEVICE_LOST when vkCmdDrawIndexedIndirectCountKHR maxDrawCount is over 1'000'000 no mater what value is in counter buffer
