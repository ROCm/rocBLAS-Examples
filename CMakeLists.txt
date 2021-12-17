
cmake_minimum_required( VERSION 3.5 )

set( CMAKE_CXX_STANDARD 17 )

project( rocblas-examples LANGUAGES CXX )

if (WIN32)
    if (NOT CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH C:/hipSDK)
    endif()
endif()

find_package(rocblas REQUIRED)

file( GLOB msvc_examples
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    # building with msvc
    ${CMAKE_SOURCE_DIR}/Level-1/*/*.cpp
    ${CMAKE_SOURCE_DIR}/Level-2/*gemv*/*.cpp
    # level 2 her using hip type
    ${CMAKE_SOURCE_DIR}/Level-3/*/*.cpp
    ${CMAKE_SOURCE_DIR}/Extensions/*gemm_ex_i*/*.cpp
    ${CMAKE_SOURCE_DIR}/Extensions/*gemm_ex_f32*/*.cpp
    # bf16 helpers require amd clang or user defined operators so skipped
    ${CMAKE_SOURCE_DIR}/Patterns/*/*.cpp
)

file( GLOB hipcc_examples
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    # either clang or hip types so using amd clang compiler
    ${CMAKE_SOURCE_DIR}/Level-2/*her*/*.cpp
    ${CMAKE_SOURCE_DIR}/Extensions/*gemm_ex_bf*/*.cpp
    ${CMAKE_SOURCE_DIR}/Extensions/*gemm_ex_f16*/*.cpp
)

add_library( examples-common STATIC ${CMAKE_SOURCE_DIR}/common/ArgParser.cpp )
target_include_directories( examples-common PRIVATE ${CMAKE_SOURCE_DIR}/common )
# other targets will inherit the rocblas dependency from this
target_link_libraries( examples-common PUBLIC roc::rocblas )

function( new_target file_name target_name )
    string(REGEX MATCH "^(.*)\/(.*)\\.[^.]*$" temp ${file_name})
    set( target_name ${CMAKE_MATCH_2})

    message(STATUS ${target_name})
    add_executable( ${target_name} ${file_name} )
    target_include_directories( ${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/common )
    set_target_properties( ${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_link_libraries( ${target_name} PRIVATE examples-common )
    set( target_name ${target_name} PARENT_SCOPE )
endfunction()

if (NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc.*")

    list( APPEND msvc_targets examples-common )

    foreach( file_i ${msvc_examples})
        new_target( ${file_i} target_name )
        list( APPEND msvc_targets ${target_name} )
    endforeach( file_i )

    message( STATUS "targets: ${msvc_targets}" )

    # msvc modifications
    foreach( target_i ${msvc_targets})
        if (WIN32)
            # required for Visual Studio or it defines it as 199711L regardless of C++ standard
            target_compile_options( ${target_i} PRIVATE /Zc:__cplusplus )
            # we use hip types so setting these 
            #target_compile_definitions( ${target_name} PRIVATE __HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__ )
        endif()
    endforeach( target_i )

else()

    list( APPEND hipcc_targets examples-common )

    foreach( file_i ${hipcc_examples})
        new_target( ${file_i} target_name )
        list( APPEND hipcc_targets ${target_name} )
    endforeach( file_i )

    message( STATUS "targets: ${hipcc_tagets}" )

    # hipcc modifications
    foreach( target_i ${hipcc_targets})
    endforeach( target_i )

    if (WIN32)
        # for now put test harness in build binary directory
        file( GLOB test_harness
            LIST_DIRECTORIES OFF
            CONFIGURE_DEPENDS
            ${CMAKE_SOURCE_DIR}/rtest.*
        )
        list(GET hipcc_targets -1 last_target)
        foreach( file_i ${test_harness})
            add_custom_command( TARGET ${last_target} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR} )
        endforeach( file_i )
        # copy msvc build executables into PROJECT_BINARY_DIR
        if (CMAKE_BUILD_TYPE MATCHES "Debug")
            set(msvc_subdir "Debug")
        else()
            set(msvc_subdir "Release")
        endif()
        add_custom_command( TARGET ${last_target} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_directory  ${PROJECT_BINARY_DIR}/../msvc/${msvc_subdir} ${PROJECT_BINARY_DIR} )
    endif()

endif()


