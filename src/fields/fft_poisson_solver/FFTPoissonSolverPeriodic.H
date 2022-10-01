/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet, Remi Lehe, WeiqunZhang
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_PERIODIC_H_
#define FFT_POISSON_SOLVER_PERIODIC_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "FFTPoissonSolver.H"
#include "utils/Constants.H"

#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>

/** Declare type for fields in spectral fields */
using SpectralField = amrex::FabArray< amrex::BaseFab <amrex::GpuComplex<amrex::Real>> >;

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
template <typename StagingType>
class FFTPoissonSolverPeriodic final : public FFTPoissonSolver<StagingType>
{
public:
    /** Constructor */
    FFTPoissonSolverPeriodic ( amrex::BoxArray const& realspace_ba,
                               amrex::DistributionMapping const& dm,
                               amrex::Geometry const& gm,
                               amrex::IntVect const nguards)
    {
        define(realspace_ba, dm, gm, nguards);
    }

    /** virtual destructor */
    virtual ~FFTPoissonSolverPeriodic () override final {}

    /**
     * \brief Define real space and spectral space boxes and multifabs, multiplier
     * coefficients inv_k2 to solve Poisson equation and FFT plans.
     * Currently only works with a single box, i.e., serial FFT.
     *
     * \param[in] realspace_ba BoxArray on which the FFT is executed.
     * \param[in] dm DistributionMapping for the BoxArray.
     * \param[in] gm Geometry, contains the box dimensions.
     */
    virtual void define ( amrex::BoxArray const& realspace_ba,
                          amrex::DistributionMapping const& dm,
                          amrex::Geometry const& gm,
                          amrex::IntVect const nguards) override final
    {
    using namespace amrex::literals;
    //using FFTPoissonSolver<StagingType>::m_spectralspace_ba;
    //using FFTPoissonSolver<StagingType>::m_stagingArea;

    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(realspace_ba.size() == 1, "Parallel FFT not supported yet");

    this->m_nguards = nguards;

    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    // Loop over boxes and fill the box list
    for (int i=0; i < realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        amrex::Box realspace_bx = realspace_ba[i];
        realspace_bx.grow(this->m_nguards);
        amrex::IntVect fft_size = realspace_bx.length();
        // Because the spectral solver uses real-to-complex FFTs, we only
        // need the positive k values along the fastest axis
        // (first axis for AMReX Fortran-order arrays) in spectral space.
        // This effectively reduces the size of the spectral space by half
        // see e.g. the FFTW documentation for real-to-complex FFTs
        amrex::IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;
        // Define the corresponding box
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          spectral_bx_size - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
    }
    this->m_spectralspace_ba.define( std::move(spectral_bl) );

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    this->m_stagingArea = amrex::MultiFab(realspace_ba, dm, 1, this->m_nguards);
    m_tmpSpectralField = SpectralField(this->m_spectralspace_ba, dm, 1, 0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(this->m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    // Calculate the array of inv_k2
    amrex::Real dkx = 2*MathConst::pi/gm.ProbLength(0);
    amrex::Real dky = 2*MathConst::pi/gm.ProbLength(1);
    m_inv_k2 = amrex::MultiFab(this->m_spectralspace_ba, dm, 1, 0);
    // Loop over boxes and calculate inv_k2 in each box
    for (amrex::MFIter mfi(m_inv_k2); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> inv_k2_arr = m_inv_k2.array(mfi);
        amrex::Box const& bx = mfi.validbox();  // The lower corner of the "2D" slice Box is zero.
        int const Ny = bx.length(1);
        int const mid_point_y = (Ny+1)/2;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
        {
            // kx is always positive (first axis of the real-to-complex FFT)
            amrex::Real kx = dkx*i;
            // The first half of ky is positive ; the other is negative
            amrex::Real ky = (j<mid_point_y) ? dky*j : dky*(j-Ny);
            if ((i!=0) && (j!=0)) {
                inv_k2_arr(i,j,0) = 1._rt/(kx*kx + ky*ky);
            } else {
                // Avoid division by 0
                inv_k2_arr(i,j,0) = 0._rt;
            }
        });
    }

    // Allocate and initialize the FFT plans
    m_forward_plan = AnyFFT::FFTplans(this->m_spectralspace_ba, dm);
    m_backward_plan = AnyFFT::FFTplans(this->m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(this->m_stagingArea); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = this->m_stagingArea[mfi].box().length();
        m_forward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, this->m_stagingArea[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::R2C);

        m_backward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, this->m_stagingArea[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( m_tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::C2R);
    }
    }

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) override final
    {
        //using FFTPoissonSolver<StagingType>::m_spectralspace_ba;
        //using FFTPoissonSolver<StagingType>::m_stagingArea;

    // Loop over boxes
    for ( amrex::MFIter mfi(this->m_stagingArea); mfi.isValid(); ++mfi ){

        // Perform Fourier transform from the staging area to `tmpSpectralField`
        AnyFFT::Execute(m_forward_plan[mfi]);

        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by inv_k2
        amrex::Array4<amrex::GpuComplex<amrex::Real>> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        amrex::Array4<amrex::Real> inv_k2_arr = m_inv_k2.array(mfi);
        amrex::ParallelFor( this->m_spectralspace_ba[mfi],
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_cmplx_arr(i,j,k) *= -inv_k2_arr(i,j,k);
            });

        // Perform Fourier transform from `tmpSpectralField` to the staging area
        AnyFFT::Execute(m_backward_plan[mfi]);

        // Copy from the staging area to output array (and normalize)
        amrex::Array4<amrex::Real> tmp_real_arr = this->m_stagingArea.array(mfi);
        amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        const amrex::Box fft_box = this->m_stagingArea[mfi].box();
        const amrex::Real inv_N = 1./fft_box.numPts();
        amrex::ParallelFor( fft_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Copy and normalize field
                lhs_arr(i,j,k) = inv_N*tmp_real_arr(i,j,k);
            });

    }
    }

private:
    /** Spectral fields, contains (complex) field in Fourier space */
    SpectralField m_tmpSpectralField;
    /** Multifab containing 1/(kx^2 + ky^2), to solve Poisson equation. */
    amrex::MultiFab m_inv_k2;
    /** FFT plans */
    AnyFFT::FFTplans m_forward_plan, m_backward_plan;
};

#endif
