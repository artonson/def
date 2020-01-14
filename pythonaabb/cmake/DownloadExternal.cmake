################################################################################
include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(EXTRA_OPTIONS "")
endif()

# Shortcut function
function(download_project_aux name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${THIRD_PARTY_DIR}/${name}
        DOWNLOAD_DIR ${THIRD_PARTY_DIR}/.cache/${name}
        QUIET
        ${EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################

function(download_pybind11)
    download_project_aux(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        085a29436a8c472caaaf7157aa644b571079bcaa
    )
endfunction()

set(EIGEN_VERSION 3.2.10 CACHE STRING "Default version of Eigen.")
function(download_eigen)
	download_project_aux(eigen
		GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
		GIT_TAG        ${EIGEN_VERSION}
	)
endfunction()
