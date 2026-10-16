// Writes a small 2D field to HDF5 so we can check the library works end to end.
#include <hdf5.h>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const hsize_t ny = 64, nx = 128;
    std::vector<double> rho(ny * nx);
    for (hsize_t j = 0; j < ny; ++j)
        for (hsize_t i = 0; i < nx; ++i)
            rho[j * nx + i] = 1.0 + 0.2 * std::sin(2.0 * M_PI * i / nx)
                                        * std::cos(2.0 * M_PI * j / ny);

    hid_t file = H5Fcreate("hdf5_check.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { std::cerr << "Could not create file\n"; return 1; }

    hsize_t dims[2] = {ny, nx};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(file, "rho", H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho.data());

    double t = 0.25;
    hid_t scalar = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, "t", H5T_NATIVE_DOUBLE, scalar, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &t);

    H5Aclose(attr); H5Sclose(scalar);
    H5Dclose(dset); H5Sclose(space);
    H5Fclose(file);

    unsigned maj, min, rel;
    H5get_libversion(&maj, &min, &rel);
    std::cout << "Wrote hdf5_check.h5 using HDF5 " << maj << '.' << min << '.' << rel << '\n';
    return 0;
}