file(GLOB BASE_SRC "*.cpp")
file(GLOB BASE_HEADERS "*.hpp")

if(WIN32)
    add_library(base STATIC ${BASE_SRC})
    target_link_libraries(base ${Vulkan_LIBRARY} ${WINLIBS})
endif(WIN32)