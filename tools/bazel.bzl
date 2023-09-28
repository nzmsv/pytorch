load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")
load("@pytorch//c10/macros:cmake_configure_file.bzl", "cmake_configure_file")
load("@pytorch//tools/config:defs.bzl", "if_cuda")

def _genrule(**kwds):
    if _enabled(**kwds):
        native.genrule(**kwds)

def _is_cpu_static_dispatch_build():
    return False

EXTRA_CPP_OPTS = [
    "-I external/pytorch",
    "-isystem bazel-out/k8-fastbuild/bin/external/pytorch",
]

def _cc_library(*args, **kwargs):
    kwargs["copts"] = EXTRA_CPP_OPTS + kwargs.get("copts", [])
    return native.cc_library(*args, **kwargs)

def _requires_cuda_enabled():
    """Returns constraint_setting that is not satisfied unless :is_cuda_enabled.
    Add to 'target_compatible_with' attribute to mark a target incompatible when
    @rules_cuda//cuda:enable_cuda is not set. Incompatible targets are excluded
    from bazel target wildcards and fail to build if requested explicitly."""
    return select({
        "@pytorch//tools/config:cuda_enabled_and_capable": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

# Rules implementation for the Bazel build system. Since the common
# build structure aims to replicate Bazel as much as possible, most of
# the rules simply forward to the Bazel definitions.
rules = struct(
    cc_binary = cc_binary,
    cc_library = _cc_library,
    cc_test = cc_test,
    cmake_configure_file = cmake_configure_file,
    cuda_library = cuda_library,
    filegroup = native.filegroup,
    genrule = _genrule,
    glob = native.glob,
    if_cuda = if_cuda,
    is_cpu_static_dispatch_build = _is_cpu_static_dispatch_build,
    py_binary = py_binary,
    py_library = py_library,
    requirement = requirement,
    requires_cuda_enabled = _requires_cuda_enabled,
    select = select,
    test_suite = native.test_suite,
)

def _enabled(tags = [], **_kwds):
    """Determines if the target is enabled."""
    return "-bazel" not in tags
