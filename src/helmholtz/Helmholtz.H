/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#ifndef Helmholtz_H_
#define Helmholtz_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>
#include <AMReX_GpuComplex.H>

/** \brief describes which slice with respect to the currently calculated is used */
namespace WhichHelmholtzSlice {
    // n stands for the time step, j for the longitudinal slice.
    // n00 is time step n, nm1 is n-1 and np1 is n+1. Similar notation for slice j.
    enum slice {
        nm1j00_r,   nm1jp1_r,   nm1jp2_r,
        n00j00_r,   n00jp1_r,   n00jp2_r,
        np1j00_r,   np1jp1_r,   np1jp2_r,
        dz_jx, dx_jz, dt_jx,
        N
    };
}

class Fields;

class MultiPlasma;

class Helmholtz
{

    using SpectralFieldLoc = amrex::BaseFab <amrex::GpuComplex<amrex::Real>>;

public:

    /** Constructor */
    explicit Helmholtz ()
    {
        ReadParameters();
    }

    void ReadParameters ();

    /** get function for the 2D slices */
    amrex::MultiFab& getSlices () {return m_slices; }

    /** get function for the 2D slices (const version) */
    const amrex::MultiFab& getSlices () const {return m_slices; }

    /** \brief Make Helmholtz geometry
     * \param[in] field_geom_3D 3D Geometry for level 0
     */
    void MakeHelmholtzGeometry (const amrex::Geometry& field_geom_3D);

    /** \brief Allocate Helmholtz multifab */
    void InitData ();

    /** \brief Initialize on slice of the 3D Helmholtz field.
     *
     * \param[in] islice slice index, referring to the 3D slice
     * \param[in] comp Helmholtz component to initialize
     */
    void InitSliceEnvelope (const int islice, const int comp);

    /** \brief Shift 2D slices in zeta
     * \param[in] islice slice index
     */
    void ShiftHelmholtzSlices (const int islice);

    /** Write Aabs into Fields MultiFab
     * \param[in] islice slice index
     * \param[in] current_N_level number of MR levels active on the current slice
     * \param[in] fields Field object
     * \param[in] field_geom Geometry of the problem
     */
    void UpdateHelmholtzAabs (const int islice, const int current_N_level, Fields& fields,
                          amrex::Vector<amrex::Geometry> const& field_geom);

    /** Put Jx from the fields into the Jx component of the Helmholtz
     * \param[in] fields Field object
     * \param[in] geom_field_lev0 Geometry of the fields on MR level 0
     */
    void InterpolateJx (const Fields& fields, amrex::Geometry const& geom_field_lev0);

    /** Wrapper function to advance a Helmholtz slice by 1 time step.
     * \param[in] islice slice index
     * \param[in] fields Field object
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     * \param[in] geom_field_lev0 Geometry of the fields on MR level 0
     */
    void AdvanceSlice (const int islice, const Fields& fields, amrex::Real dt, int step,
                       amrex::Geometry const& geom_field_lev0);

    /** Advance a Helmholtz slice by 1 time step using a FFT solver.
     * The complex phase of the envelope is evaluated on-axis only.
     *
     * \param[in] dt time step of the simulation
     * \param[in] step current iteration. Needed because step 0 needs a specific treatment.
     */
    void AdvanceSliceFFT (amrex::Real dt, int step);

    /** Initialize 1 longitudinal slice of the Helmholtz, and store it in n00j00 (current time step)
     * and nm1j00 (previous time step).
     *
     * \param[in] comp Helmholtz component to initialize
     */
    void InitHelmholtzSlice (const int comp);

    /** Compute in-situ Helmholtz diagnostics of current slice, store in member variable
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] islice current slice, on which diags are computed.
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituComputeDiags (int step, amrex::Real time, int islice,
                             int max_step, amrex::Real max_time);

    /** Dump in-situ reduced diagnostics to file.
     * \param[in] step current time step
     * \param[in] time physical time
     * \param[in] max_step maximum time step of simulation
     * \param[in] max_time maximum time of simulation
     */
    void InSituWriteToFile (int step, amrex::Real time, int max_step, amrex::Real max_time);

    /** Get the geometry of the Helmholtz Box */
    const amrex::Geometry& GetHelmholtzGeom () const { return m_helmholtz_geom_3D; }

    /** If the Helmholtz geometry includes this slice
     * \param[in] islice slice index
     */
    bool HasSlice (const int islice) const {
        return GetHelmholtzGeom().Domain().smallEnd(2) <= islice &&
               islice <= GetHelmholtzGeom().Domain().bigEnd(2);
    }

    /** If the Helmholtz is used */
    bool UseHelmholtz () const { return m_use_helmholtz; }

    /** Whether the Helmholtz is used and the Helmholtz geometry includes this slice
     * \param[in] islice slice index
     */
    bool UseHelmholtz (const int islice) const { return m_use_helmholtz && HasSlice(islice); }

    bool UseDxJz () const { return m_use_helmholtz && m_use_dx_jz; }

private:

    bool m_use_helmholtz {false}; /**< whether a helmholtz is used or not */
    bool m_use_dx_jz {true};
    bool m_use_dz_jx {false};
    bool m_use_dt_jx {false};
    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards = {-1, -1, -1};
    /** 3D helmholtz Geometry */
    amrex::Geometry m_helmholtz_geom_3D;
    /** xy slice BoxArray. Contains only one box */
    amrex::BoxArray m_helmholtz_slice_ba;
    /** xy slice DistributionMapping */
    amrex::DistributionMapping m_helmholtz_slice_dm;
    /** slice box of helmholtz  */
    amrex::Box m_slice_box;
    /** interpolation order for helmholtz to field and field to helmholtz operations */
    int m_interp_order = 1;

    /** Array of N slices required to compute current slice */
    amrex::MultiFab m_slices;

    /** FFTW plan for forward C2C transform to solve Complex Poisson equation */
    AnyFFT m_forward_fft;
    /** FFTW plan for backward C2C transform to solve Complex Poisson equation */
    AnyFFT m_backward_fft;
    /** work area for both FFT plans */
    amrex::Gpu::DeviceVector<char> m_fft_work_area;
    /** Complex FAB to store the solution (e.g. helmholtz envelope on current slice) */
    SpectralFieldLoc m_sol;
    /** Complex FAB to store the RHS in position space */
    SpectralFieldLoc m_rhs;
    /** Complex FAB to store the RHS in Fourier space */
    SpectralFieldLoc m_rhs_fourier;

    // Data for in-situ diagnostics:
    /** Number of real helmholtz properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_nrp = 6;
    /** Number of real complex properties for in-situ per-slice reduced diagnostics. */
    static constexpr int m_insitu_ncp = 1;
    /** How often the insitu helmholtz diagnostics should be computed and written
     * Default is 0, meaning no output */
    int m_insitu_period {0};
    /** All per-slice real helmholtz properties */
    amrex::Vector<amrex::Real> m_insitu_rdata;
    /** Sum of all per-slice real helmholtz properties */
    amrex::Vector<amrex::Real> m_insitu_sum_rdata;
    /** All per-slice complex helmholtz properties */
    amrex::Vector<amrex::GpuComplex<amrex::Real>> m_insitu_cdata;
    /** Prefix/path for the output files */
    std::string m_insitu_file_prefix = "diags/helmholtz_insitu";
};

#endif // Helmholtz_H_
