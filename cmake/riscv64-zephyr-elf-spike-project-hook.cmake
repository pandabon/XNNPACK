# Loaded via CMAKE_PROJECT_INCLUDE from riscv64-zephyr-elf-spike.toolchain.
# CMAKE_PROJECT_INCLUDE fires after EVERY project() call, including the
# tiny ones inside try_compile probes and inside add_subdirectory'd
# dependencies (googletest, pthreadpool, fxdiv). Self-skip outside the
# top-level XNNPACK project, and stay idempotent within it.

IF(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "XNNPACK")
  RETURN()
ENDIF()
IF(TARGET htif-runtime)
  RETURN()
ENDIF()

# googletest's install/export rules trip when we propagate htif-runtime
# via link_libraries(); we aren't installing this build anyway.
SET(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Empty Threads::Threads so XNNPack's find_package(Threads) sees it
# already exists and skips its -lpthreads probe.
IF(NOT TARGET Threads::Threads)
  ADD_LIBRARY(Threads::Threads INTERFACE IMPORTED GLOBAL)
ENDIF()

# Build the self-contained HTIF runtime.
ADD_LIBRARY(htif-runtime STATIC
    "${HTIF_DIR}/crt.S"
    "${HTIF_DIR}/syscalls.c")
TARGET_INCLUDE_DIRECTORIES(htif-runtime PRIVATE "${HTIF_DIR}")

# Every executable links against the runtime. ADD_EXECUTABLE() inherits
# link_libraries() set in the directory scope; this directory is the
# top-level XNNPack source dir, so all subdirectories see it.
LINK_LIBRARIES(htif-runtime)
