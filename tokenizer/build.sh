export CIBW_MANYLINUX_X86_64_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
export CIBW_BUILD="cp310-* cp311-* cp312-* cp313-*"

export CIBW_BEFORE_ALL_LINUX="dnf install -y CGAL-devel && dnf clean all"

python -m cibuildwheel --output-dir dist
