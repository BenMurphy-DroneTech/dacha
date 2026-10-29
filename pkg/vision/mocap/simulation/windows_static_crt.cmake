# Windows builds of mocap_app with --features mocap_manager/simulation: dacha
# links Windows binaries with the static MSVC runtime (+crt-static), but
# glfw-sys builds GLFW against the DLL runtime unless told otherwise. Point
# CMAKE_TOOLCHAIN_FILE_x86_64_pc_windows_msvc at this file (with CMake on PATH,
# e.g. the copy inside Visual Studio's Build Tools).
set(USE_MSVC_RUNTIME_LIBRARY_DLL OFF CACHE BOOL "Static MSVC runtime, as dacha's +crt-static" FORCE)
