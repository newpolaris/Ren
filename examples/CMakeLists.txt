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
    if(WIN32)
        add_executable(${EXAMPLE_NAME} ${MAIN_CPP} ${SOURCE} ${SHADERS})
        target_link_libraries(${EXAMPLE_NAME} base meshoptimizer ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} ${WINLIBS} glfw ${GLFW_LIBRARIES})
    else(WIN32)
        add_executable(${EXAMPLE_NAME} ${MAIN_CPP} ${SOURCE} ${SHADERS})
        target_link_libraries(${EXAMPLE_NAME} base meshoptimizer)
    endif(WIN32)

    set_target_properties(${EXAMPLE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

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
        helloworld
)

buildExamples()
