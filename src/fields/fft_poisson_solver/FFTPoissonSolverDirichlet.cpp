/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichlet.H"
#include "fft/AnyDST.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichlet::FFTPoissonSolverDirichlet (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm,
    amrex::IntVect const nguards)
{
    define(realspace_ba, dm, gm, nguards);
}

void
FFTPoissonSolverDirichlet::define (amrex::BoxArray const& a_realspace_ba,
                                   amrex::DistributionMapping const& dm,
                                   amrex::Geometry const& gm,
                                   amrex::IntVect const nguards)
{
    using namespace amrex::literals;

    m_nguards = nguards;

    HIPACE_PROFILE("FFTPoissonSolverDirichlet::define()");
    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    m_spectralspace_ba = a_realspace_ba;

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(a_realspace_ba, dm, 1, m_nguards);
    m_tmpSpectralField = amrex::MultiFab(a_realspace_ba, dm, 1, m_nguards);
    m_eigenvalue_matrix = amrex::MultiFab(a_realspace_ba, dm, 1, m_nguards);
    m_stagingArea.setVal(0.0, m_nguards); // this is not required
    m_tmpSpectralField.setVal(0.0, m_nguards);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    const amrex::Box fft_box = m_stagingArea[0].box();
    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( fft_box.length(0) + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( fft_box.length(1) + 1 ));

    // Normalization of FFTW's 'DST-I' discrete sine transform (FFTW_RODFT00)
    // This normalization is used regardless of the sine transform library
    const amrex::Real norm_fac = 0.5 / ( 2 * (( fft_box.length(0) + 1 )
                                             *( fft_box.length(1) + 1 )));

    // Calculate the array of m_eigenvalue_matrix
    for (amrex::MFIter mfi(m_eigenvalue_matrix); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::IntVect lo = fft_box.smallEnd();
        amrex::ParallelFor(
            fft_box, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
                {
                    /* fast poisson solver diagonal x coeffs */
                    amrex::Real sinex_sq = sin(( i - lo[0] + 1 ) * sine_x_factor) * sin(( i - lo[0] + 1 ) * sine_x_factor);
                    /* fast poisson solver diagonal y coeffs */
                    amrex::Real siney_sq = sin(( j - lo[1] + 1 ) * sine_y_factor) * sin(( j - lo[1] + 1 ) * sine_y_factor);

                    if ((sinex_sq!=0) && (siney_sq!=0)) {
                        eigenvalue_matrix(i,j,lo[2]) = norm_fac / ( -4.0 * ( sinex_sq / dxsquared + siney_sq / dysquared ));
                    } else {
                        // Avoid division by 0
                        eigenvalue_matrix(i,j,lo[2]) = 0._rt;
                    }
                });
    }

    // Allocate and initialize the FFT plans
    m_plan = AnyDST::DSTplans(m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = fft_box.length();
        m_plan[mfi] = AnyDST::CreatePlan(
            fft_size, &m_stagingArea[mfi], &m_tmpSpectralField[mfi]);
    }
}


void
FFTPoissonSolverDirichlet::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::SolvePoissonEquation()");

    // Loop over boxes
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){

        // Perform Fourier transform from the staging area to `tmpSpectralField`
        AnyDST::Execute<AnyDST::direction::forward>(m_plan[mfi]);

        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        amrex::Array4<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( m_tmpSpectralField[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_cmplx_arr(i,j,k) *= eigenvalue_matrix(i,j,k);
            });

        // Perform Fourier transform from `tmpSpectralField` to the staging area
        AnyDST::Execute<AnyDST::direction::backward>(m_plan[mfi]);

        // Copy from the staging area to output array (and normalize)
        amrex::Array4<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lhs_mf.size() == 1,
                                         "Slice MFs must be defined on one box only");
        amrex::ParallelFor( lhs_mf[mfi].box() & m_stagingArea[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Copy field
                lhs_arr(i,j,k) = tmp_real_arr(i,j,k);
            });
    }
}
