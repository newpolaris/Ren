if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    SET(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
    SET(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

# Function for building single example
function(buildExample EXAMPLE_NAME)
    SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME})
    message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
    # Main
    file(GLOB SOURCE *.cpp ${BASE_HEADER} ${EXAMPLE_FOLDER}/*.cpp)
    SET(MAIN_CPP ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)
    if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
        SET(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
    ENDIF()

	# Add shaders
    SET(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders/${EXAMPLE_NAME}")
    file(GLOB SHADER_HEADERS "${SHADER_DIR}/*.h")
	file(GLOB SHADERS "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp" "${SHADER_DIR}/*.geom" "${SHADER_DIR}/*.tesc" "${SHADER_DIR}/*.tese" "${SHADER_DIR}/*.mesh")
    source_group("Shaders" FILES ${SHADERS} ${SHADER_HEADERS})

    foreach(GLSL ${SHADERS})
        get_filename_component(FILE_NAME ${GLSL} NAME)
        SET(SPIRV "${SHADER_DIR}/${FILE_NAME}.spv")
        add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_DIR}"
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL} ${SHADER_HEADERS})
        list(APPEND SPIRV_BINARY_FILES ${SPIRV})
    endforeach(GLSL)

    SET(EXAMPLE_RESOURCE "${EXAMPLE_NAME}-resource")
    add_custom_target(${EXAMPLE_RESOURCE} DEPENDS ${SPIRV_BINARY_FILES})

    if(WIN32)
        add_executable(${EXAMPLE_NAME} ${MAIN_CPP} ${SOURCE} ${SHADERS} ${SHADER_HEADERS})
        target_link_libraries(${EXAMPLE_NAME} base meshoptimizer volk ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} ${WINLIBS} glfw ${GLFW_LIBRARIES})
    else(WIN32)
        add_executable(${EXAMPLE_NAME} ${MAIN_CPP} ${SOURCE} ${SHADERS})
        target_link_libraries(${EXAMPLE_NAME} base meshoptimizer volk ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} ${WINLIBS} glfw ${GLFW_LIBRARIES})
    endif(WIN32)
    add_dependencies(${EXAMPLE_NAME} ${EXAMPLE_RESOURCE})

    set_target_properties(${EXAMPLE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(RESOURCE_INSTALL_DIR)
        install(TARGETS ${EXAMPLE_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endfunction(buildExample)

# Build all examples
function(buildExamples)
    foreach(EXAMPLE ${EXAMPLES})
        buildExample(${EXAMPLE})
    endforeach(EXAMPLE)
endfunction(buildExamples)

set (EXAMPLES
        01.helloworld
        02.modelRendering
        03.pushDescriptors
        04.meshlet
        05.rin
        06.spirv-reflect
)

buildExamples()
