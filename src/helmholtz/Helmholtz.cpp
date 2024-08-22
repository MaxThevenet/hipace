/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#include "Helmholtz.H"
#include "utils/Constants.H"
#include "fields/Fields.H"
#include "Hipace.H"
#include "particles/plasma/MultiPlasma.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/InsituUtil.H"
#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "particles/particles_utils/ShapeFactors.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

#include <AMReX_GpuComplex.H>

void
Helmholtz::ReadParameters ()
{
    amrex::ParmParse pp("helmholtzs");

    queryWithParser(pp, "use_helmholtz", m_use_helmholtz);
    queryWithParser(pp, "use_jz_correction", m_use_jz_correction);
    queryWithParser(pp, "use_dt_jx", m_use_dt_jx);

    if (!m_use_helmholtz) return;

    queryWithParser(pp, "interp_order", m_interp_order);
    AMREX_ALWAYS_ASSERT(m_interp_order <= 3 && m_interp_order >= 0);
    queryWithParser(pp, "insitu_period", m_insitu_period);
    queryWithParser(pp, "insitu_file_prefix", m_insitu_file_prefix);
}

void
Helmholtz::MakeHelmholtzGeometry (const amrex::Geometry& field_geom_3D)
{
    if (!m_use_helmholtz) return;
    amrex::ParmParse pp("helmholtzs");

    // use field_geom_3D as the default
    std::array<int, 2> n_cells_helmholtz {field_geom_3D.Domain().length(0),
                                          field_geom_3D.Domain().length(1)};
    std::array<amrex::Real, 3> patch_lo_helmholtz {
        field_geom_3D.ProbDomain().lo(0),
        field_geom_3D.ProbDomain().lo(1),
        field_geom_3D.ProbDomain().lo(2)};
    std::array<amrex::Real, 3> patch_hi_helmholtz {
        field_geom_3D.ProbDomain().hi(0),
        field_geom_3D.ProbDomain().hi(1),
        field_geom_3D.ProbDomain().hi(2)};

    // get parameters from user input
    queryWithParser(pp, "n_cell", n_cells_helmholtz);
    queryWithParser(pp, "patch_lo", patch_lo_helmholtz);
    queryWithParser(pp, "patch_hi", patch_hi_helmholtz);

    // round zeta lo and hi to full cells
    const amrex::Real pos_offset_z = GetPosOffset(2, field_geom_3D, field_geom_3D.Domain());

    const int zeta_lo = std::max( field_geom_3D.Domain().smallEnd(2),
        int(amrex::Math::round((patch_lo_helmholtz[2] - pos_offset_z) * field_geom_3D.InvCellSize(2)))
    );

    const int zeta_hi = std::min( field_geom_3D.Domain().bigEnd(2),
        int(amrex::Math::round((patch_hi_helmholtz[2] - pos_offset_z) * field_geom_3D.InvCellSize(2)))
    );

    patch_lo_helmholtz[2] = (zeta_lo-0.5)*field_geom_3D.CellSize(2) + pos_offset_z;
    patch_hi_helmholtz[2] = (zeta_hi+0.5)*field_geom_3D.CellSize(2) + pos_offset_z;

    // make the boxes
    const amrex::Box domain_3D_helmholtz{amrex::IntVect(0, 0, zeta_lo),
        amrex::IntVect(n_cells_helmholtz[0]-1, n_cells_helmholtz[1]-1, zeta_hi)};

    const amrex::RealBox real_box(patch_lo_helmholtz, patch_hi_helmholtz);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(real_box.volume() > 0., "Helmholtz box must have positive volume");

    // make the geometry, slice box and ba and dm
    m_helmholtz_geom_3D.define(domain_3D_helmholtz, real_box, amrex::CoordSys::cartesian, {0, 0, 0});

    m_slice_box = domain_3D_helmholtz;
    m_slice_box.setSmall(2, 0);
    m_slice_box.setBig(2, 0);

    m_helmholtz_slice_ba.define(m_slice_box);
    m_helmholtz_slice_dm.define(amrex::Vector<int>({amrex::ParallelDescriptor::MyProc()}));
}

void
Helmholtz::InitData ()
{
    if (!m_use_helmholtz) return;

    HIPACE_PROFILE("Helmholtz::InitData()");

    // Alloc 2D slices
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = (Hipace::m_depos_order_xy + 1) / 2 + 1;
    m_slices_nguards = {nguards_xy, nguards_xy, 0};
    m_slices.define(
        m_helmholtz_slice_ba, m_helmholtz_slice_dm, WhichHelmholtzSlice::N, m_slices_nguards,
        amrex::MFInfo().SetArena(amrex::The_Arena()));
    m_slices.setVal(0.0);

    m_sol.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs.resize(m_slice_box, 1, amrex::The_Arena());
    m_rhs_fourier.resize(m_slice_box, 1, amrex::The_Arena());

    // Create FFT plans
    amrex::IntVect fft_size = m_slice_box.length();

    std::size_t fwd_area = m_forward_fft.Initialize(FFTType::C2C_2D_fwd, fft_size[0], fft_size[1]);
    std::size_t bkw_area = m_backward_fft.Initialize(FFTType::C2C_2D_bkw, fft_size[0], fft_size[1]);

    // Allocate work area for both FFTs
    m_fft_work_area.resize(std::max(fwd_area, bkw_area));

    m_forward_fft.SetBuffers(m_rhs.dataPtr(), m_rhs_fourier.dataPtr(), m_fft_work_area.dataPtr());
    m_backward_fft.SetBuffers(m_rhs_fourier.dataPtr(), m_sol.dataPtr(), m_fft_work_area.dataPtr());

    if (m_insitu_period > 0) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different field insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_insitu_rdata.resize(m_helmholtz_geom_3D.Domain().length(2)*m_insitu_nrp, 0.);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
        m_insitu_cdata.resize(m_helmholtz_geom_3D.Domain().length(2)*m_insitu_ncp, 0.);
    }
}

void
Helmholtz::InitSliceEnvelope (const int islice, const int comp)
{
    if (!UseHelmholtz(islice)) return;

    HIPACE_PROFILE("Helmholtz::InitSliceEnvelope()");

    InitHelmholtzSlice(comp);
}

void
Helmholtz::ShiftHelmholtzSlices (const int islice)
{
    if (!UseHelmholtz(islice)) return;

    HIPACE_PROFILE("Helmholtz::ShiftHelmholtzSlices()");

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        const amrex::Box bx = mfi.tilebox();
        Array3<amrex::Real> arr = m_slices.array(mfi);
        amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
        {
            using namespace WhichHelmholtzSlice;
            // 2 components for complex numbers.
            // Shift slices of step n-1
            const amrex::Real tmp_nm1j00 = arr(i, j, nm1jp2_r);
            arr(i, j, nm1jp2_r) = arr(i, j, nm1jp1_r);
            arr(i, j, nm1jp1_r) = arr(i, j, nm1j00_r);
            arr(i, j, nm1j00_r) = tmp_nm1j00;
            // Shift slices of step n
            const amrex::Real tmp_n00j00 = arr(i, j, n00jp2_r);
            arr(i, j, n00jp2_r) = arr(i, j, n00jp1_r);
            arr(i, j, n00jp1_r) = arr(i, j, n00j00_r);
            arr(i, j, n00j00_r) = tmp_n00j00;
            // Shift slices of step n+1
            arr(i, j, np1jp2_r) = arr(i, j, np1jp1_r);
            arr(i, j, np1jp1_r) = arr(i, j, np1j00_r);
            // np1j00_r will be computed by AdvanceSlice
        });
    }
}

void
Helmholtz::InterpolateJx (const Fields& fields, amrex::Geometry const& geom_field_lev0)
{
    HIPACE_PROFILE("Helmholtz::InterpolateJx()");

    using namespace amrex::literals;

    const bool use_jz_correction = m_use_jz_correction;

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        Array3<amrex::Real> helmholtz_arr = m_slices.array(mfi);
        Array3<const amrex::Real> field_arr = fields.getSlices(0).array(mfi);

        const amrex::Real dx_inv = m_helmholtz_geom_3D.InvCellSize(0);
        const amrex::Real dz_inv = m_helmholtz_geom_3D.InvCellSize(2);

        const int jz_this = Comps[WhichSlice::This]["jz_beam"];
        const int jx_this = Comps[WhichSlice::This]["jx_beam"];
        const int jx_prev = Comps[WhichSlice::Previous]["jx_beam"];
        const int jx_prev2 = Comps[WhichSlice::Previous2]["jx_beam"];
        const int dt_jx_this = Comps[WhichSlice::This]["jy_beam"];

        const amrex::Real poff_helmholtz_x = GetPosOffset(0, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
        const amrex::Real poff_helmholtz_y = GetPosOffset(1, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
        const amrex::Real poff_field_x = GetPosOffset(0, geom_field_lev0, geom_field_lev0.Domain());
        const amrex::Real poff_field_y = GetPosOffset(1, geom_field_lev0, geom_field_lev0.Domain());

        const amrex::Real dx_helmholtz = m_helmholtz_geom_3D.CellSize(0);
        const amrex::Real dy_helmholtz = m_helmholtz_geom_3D.CellSize(1);
        const amrex::Real dx_helmholtz_inv = m_helmholtz_geom_3D.InvCellSize(0);
        const amrex::Real dy_helmholtz_inv = m_helmholtz_geom_3D.InvCellSize(1);
        const amrex::Real dx_field = geom_field_lev0.CellSize(0);
        const amrex::Real dy_field = geom_field_lev0.CellSize(1);
        const amrex::Real dx_field_inv = geom_field_lev0.InvCellSize(0);
        const amrex::Real dy_field_inv = geom_field_lev0.InvCellSize(1);

        const amrex::Box field_box = fields.getSlices(0)[mfi].box();

        const amrex::Real pos_x_lo = field_box.smallEnd(0) * dx_field + poff_field_x;
        const amrex::Real pos_x_hi = field_box.bigEnd(0) * dx_field + poff_field_x;
        const amrex::Real pos_y_lo = field_box.smallEnd(1) * dy_field + poff_field_y;
        const amrex::Real pos_y_hi = field_box.bigEnd(1) * dy_field + poff_field_y;

        // the indexes of the helmholtz box where the fields box ends
        const int x_lo = amrex::Math::ceil((pos_x_lo - poff_helmholtz_x) * dx_helmholtz_inv);
        const int x_hi = amrex::Math::floor((pos_x_hi - poff_helmholtz_x) * dx_helmholtz_inv);
        const int y_lo = amrex::Math::ceil((pos_y_lo - poff_helmholtz_y) * dy_helmholtz_inv);
        const int y_hi = amrex::Math::floor((pos_y_hi - poff_helmholtz_y) * dy_helmholtz_inv);

        amrex::ParallelFor(
            amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
            {m_interp_order},
            mfi.growntilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int, auto interp_order) noexcept {
                const amrex::Real x = i * dx_helmholtz + poff_helmholtz_x;
                const amrex::Real y = j * dy_helmholtz + poff_helmholtz_y;

                const amrex::Real xmid = (x - poff_field_x) * dx_field_inv;
                const amrex::Real ymid = (y - poff_field_y) * dy_field_inv;

                constexpr int derivative_type = 1;

                amrex::Real dx_jz = 0._rt;
                amrex::Real jx_t = 0._rt;
                amrex::Real jx_p = 0._rt;
                amrex::Real jx_p2 = 0._rt;
                amrex::Real dt_jx = 0._rt;

                if (x_lo <= i && i <= x_hi && y_lo <= j && j <= y_hi) {
                    // interpolate jx from fields to helmholtz
                    for (int iy=0; iy<=interp_order; ++iy) {
                        for (int ix=0; ix<=interp_order; ++ix) {
                            auto [shape_x, cell_x] =
                                compute_single_shape_factor<false, interp_order>(xmid, ix);
                            auto [shape_y, cell_y] =
                                compute_single_shape_factor<false, interp_order>(ymid, iy);
                            jx_t  += shape_x*shape_y*field_arr(cell_x, cell_y, jx_this);
                            jx_p  += shape_x*shape_y*field_arr(cell_x, cell_y, jx_prev);
                            jx_p2 += shape_x*shape_y*field_arr(cell_x, cell_y, jx_prev2);
                            dt_jx += shape_x*shape_y*field_arr(cell_x, cell_y, dt_jx_this);
                        }
                    }
                }
                helmholtz_arr(i, j, WhichHelmholtzSlice::dz_jx) =
                    ( -3._rt*jx_t + 4._rt*jx_p - jx_p2 ) * 0.5_rt * dz_inv;
                 helmholtz_arr(i, j, WhichHelmholtzSlice::dt_jx) = dt_jx;

                if (!use_jz_correction) return;

                if (x_lo + derivative_type <= i && i <= x_hi - derivative_type && y_lo <= j && j <= y_hi) {
                    for (int iy=0; iy<=interp_order; ++iy) {
                        for (int ix=0; ix<=interp_order + derivative_type; ++ix) {
                            auto [shape_x, shape_dx, cell_x] =
                                single_derivative_shape_factor<derivative_type, interp_order>(xmid, ix);
                            auto [shape_y, cell_y] =
                                compute_single_shape_factor<false, interp_order>(ymid, iy);
                            dx_jz += shape_dx*shape_y*field_arr(cell_x, cell_y, jz_this);
                        }
                    }
                }
                helmholtz_arr(i, j, WhichHelmholtzSlice::dx_jz) = dx_jz * dx_inv;
            });
    }
}

void
Helmholtz::AdvanceSlice (const int islice, const Fields& fields, amrex::Real dt, int step,
                          amrex::Geometry const& geom_field_lev0)
{

    if (!UseHelmholtz(islice)) return;

    InterpolateJx(fields, geom_field_lev0);

    AdvanceSliceFFT(dt, step);
}

void
Helmholtz::AdvanceSliceFFT (const amrex::Real dt, int step)
{

    HIPACE_PROFILE("Helmholtz::AdvanceSliceFFT()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    const bool use_jz_correction = m_use_jz_correction;
    const bool use_dt_jx = m_use_dt_jx;

    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);
    const amrex::Real dz = m_helmholtz_geom_3D.CellSize(2);

    const PhysConst phc = get_phys_const();
    const amrex::Real c = phc.c;

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        const int imin = bx.smallEnd(0);
        const int imax = bx.bigEnd  (0);
        const int jmin = bx.smallEnd(1);
        const int jmax = bx.bigEnd  (1);

        // solution: complex array
        // The right-hand side is computed and stored in rhs
        // Then rhs is Fourier-transformed into rhs_fourier, then multiplied by -1/(k**2+a)
        // rhs_fourier is FFT-back-transformed to sol, and sol is normalized and copied into np1j00.
        Array3<Complex> sol_arr = m_sol.array();
        Array3<Complex> rhs_arr = m_rhs.array();
        amrex::Array4<Complex> rhs_fourier_arr = m_rhs_fourier.array();

        Array3<amrex::Real> arr = m_slices.array(mfi);

        int const Nx = bx.length(0);
        int const Ny = bx.length(1);

        // Get the central point. Useful to get the on-axis phase and calculate kx and ky.
        int const imid = (Nx+1)/2;
        int const jmid = (Ny+1)/2;

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                using namespace WhichHelmholtzSlice;
                // Transverse Laplacian of A_j^n-1
                amrex::Real lap;
                if (step == 0) {
                    lap = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, n00j00_r)+arr(i-1, j, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dx*dx) +
                        (arr(i, j+1, n00j00_r)+arr(i, j-1, n00j00_r)-2._rt*arr(i, j, n00j00_r))/(dy*dy) : 0._rt;
                } else {
                    lap = i>imin && i<imax && j>jmin && j<jmax ?
                        (arr(i+1, j, nm1j00_r)+arr(i-1, j, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dx*dx) +
                        (arr(i, j+1, nm1j00_r)+arr(i, j-1, nm1j00_r)-2._rt*arr(i, j, nm1j00_r))/(dy*dy) : 0._rt;
                }
                const amrex::Real an00j00 = arr(i, j, n00j00_r);
                const amrex::Real anp1jp1 = arr(i, j, np1jp1_r);
                const amrex::Real anp1jp2 = arr(i, j, np1jp2_r);
                amrex::Real rhs;
                if (step == 0) {
                    // First time step: non-centered push to go
                    // from step 0 to step 1 without knowing -1.
                    const amrex::Real an00jp1 = arr(i, j, n00jp1_r);
                    const amrex::Real an00jp2 = arr(i, j, n00jp2_r);
                    rhs =
                        + 8._rt/(c*dt*dz)*(-anp1jp1+an00jp1)
                        + 2._rt/(c*dt*dz)*(+anp1jp2-an00jp2)
                        - lap
                        + ( -6._rt/(c*dt*dz) ) * an00j00;
                } else {
                    const amrex::Real anm1jp1 = arr(i, j, nm1jp1_r);
                    const amrex::Real anm1jp2 = arr(i, j, nm1jp2_r);
                    const amrex::Real anm1j00 = arr(i, j, nm1j00_r);
                    rhs =
                        + 4._rt/(c*dt*dz)*(-anp1jp1+anm1jp1)
                        + 1._rt/(c*dt*dz)*(+anp1jp2-anm1jp2)
                        - 4._rt/(c*c*dt*dt)*an00j00
                        - lap
                        + ( -3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt) ) * anm1j00;
                }
                if (use_dt_jx) {
                    rhs += 2._rt * phc.mu0 * arr(i, j, dt_jx) / dt;
                } else
                {
                    rhs -= 2._rt * phc.mu0 * c * arr(i, j, dz_jx);
                    if (use_jz_correction) {
                        rhs += 2._rt * phc.mu0 * c * arr(i, j, dx_jz); // From Ex equation
                    }
                }
                rhs_arr(i,j,0) = rhs;
            });

        // Transform rhs to Fourier space
        m_forward_fft.Execute();

        // Multiply by appropriate factors in Fourier space
        amrex::Real dkx = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(0);
        amrex::Real dky = 2.*MathConst::pi/m_helmholtz_geom_3D.ProbLength(1);
        const amrex::Real acoeff = step == 0 ? 6._rt/(c*dt*dz) : 3._rt/(c*dt*dz) + 2._rt/(c*c*dt*dt);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // divide rhs_fourier by -(k^2+a)
                amrex::Real kx = (i<imid) ? dkx*i : dkx*(i-Nx);
                amrex::Real ky = (j<jmid) ? dky*j : dky*(j-Ny);
                const amrex::Real inv_k2a = std::abs(kx*kx + ky*ky + acoeff) > 0. ? 1._rt/(kx*kx + ky*ky + acoeff) : 0.;
                rhs_fourier_arr(i,j,k,0) *= -inv_k2a;
            });

        // Transform rhs to Fourier space to get solution in sol
        m_backward_fft.Execute();

        // Normalize and store solution in np1j00[0]. Guard cells are filled with 0s.
        amrex::Box grown_bx = bx;
        grown_bx.grow(m_slices_nguards);
        const amrex::Real inv_numPts = 1./bx.numPts();
        amrex::ParallelFor(
            grown_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                using namespace WhichHelmholtzSlice;
                if (i>=imin && i<=imax && j>=jmin && j<=jmax) {
                    arr(i, j, np1j00_r) = sol_arr(i,j,0).real() * inv_numPts;
                } else {
                    arr(i, j, np1j00_r) = 0._rt;
                }
            });
    }
}

void
Helmholtz::InitHelmholtzSlice (const int comp)
{
    HIPACE_PROFILE("Helmholtz::InitHelmholtzSlice()");

    using namespace amrex::literals;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_slices, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const & arr = m_slices.array(mfi);
        // Initialize a Gaussian helmholtz envelope on slice islice
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                arr(i, j, k, comp ) = 0._rt;
            });
    }
}

void
Helmholtz::InSituComputeDiags (int step, amrex::Real time, int islice,
                                int max_step, amrex::Real max_time)
{
    if (!UseHelmholtz(islice)) return;
    if (!utils::doDiagnostics(m_insitu_period, step, max_step, time, max_time)) return;
    HIPACE_PROFILE("Helmholtz::InSituComputeDiags()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_sum_rdata.size()>0 &&
                        m_insitu_cdata.size()>0);

    const int nslices = m_helmholtz_geom_3D.Domain().length(2);
    const int helmholtz_slice = islice - m_helmholtz_geom_3D.Domain().smallEnd(2);
    const amrex::Real poff_x = GetPosOffset(0, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real poff_y = GetPosOffset(1, m_helmholtz_geom_3D, m_helmholtz_geom_3D.Domain());
    const amrex::Real dx = m_helmholtz_geom_3D.CellSize(0);
    const amrex::Real dy = m_helmholtz_geom_3D.CellSize(1);
    const amrex::Real dxdydz = dx * dy * m_helmholtz_geom_3D.CellSize(2);

    const int xmid_lo = m_helmholtz_geom_3D.Domain().smallEnd(0) + (m_helmholtz_geom_3D.Domain().length(0) - 1) / 2;
    const int xmid_hi = m_helmholtz_geom_3D.Domain().smallEnd(0) + (m_helmholtz_geom_3D.Domain().length(0)) / 2;
    const int ymid_lo = m_helmholtz_geom_3D.Domain().smallEnd(1) + (m_helmholtz_geom_3D.Domain().length(1) - 1) / 2;
    const int ymid_hi = m_helmholtz_geom_3D.Domain().smallEnd(1) + (m_helmholtz_geom_3D.Domain().length(1)) / 2;
    const amrex::Real mid_factor = (xmid_lo == xmid_hi ? 1._rt : 0.5_rt)
                                 * (ymid_lo == ymid_hi ? 1._rt : 0.5_rt);

    amrex::TypeMultiplier<amrex::ReduceOps, amrex::ReduceOpMax, amrex::ReduceOpSum[m_insitu_nrp-1+m_insitu_ncp]> reduce_op;
    amrex::TypeMultiplier<amrex::ReduceData, amrex::Real[m_insitu_nrp], Complex[m_insitu_ncp]> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for ( amrex::MFIter mfi(m_slices, DfltMfi); mfi.isValid(); ++mfi ) {
        Array3<amrex::Real const> const arr = m_slices.const_array(mfi);
        reduce_op.eval(
            mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int) -> ReduceTuple
            {
                using namespace WhichHelmholtzSlice;
                const amrex::Real areal = arr(i,j, n00j00_r);
                const amrex::Real aimag = 0._rt; // arr(i,j, n00j00_i);
                const amrex::Real aabssq = abssq(areal, aimag);

                const amrex::Real x = i * dx + poff_x;
                const amrex::Real y = j * dy + poff_y;

                const bool is_on_axis = (i==xmid_lo || i==xmid_hi) && (j==ymid_lo || j==ymid_hi);
                const Complex aaxis{is_on_axis ? areal : 0._rt, is_on_axis ? aimag : 0._rt};

                return {            // Tuple contains:
                    aabssq,         // 0    max(|a|^2)
                    aabssq,         // 1    [|a|^2]
                    aabssq*x,       // 2    [|a|^2*x]
                    aabssq*x*x,     // 3    [|a|^2*x*x]
                    aabssq*y,       // 4    [|a|^2*y]
                    aabssq*y*y,     // 5    [|a|^2*y*y]
                    aaxis           // 6    axis(a)
                };
            });
    }

    ReduceTuple a = reduce_data.value();

    amrex::constexpr_for<0, m_insitu_nrp>(
        [&] (auto idx) {
            if (idx == 0) {
                m_insitu_rdata[helmholtz_slice + idx * nslices] = amrex::get<idx>(a);
                m_insitu_sum_rdata[idx] = std::max(m_insitu_sum_rdata[idx], amrex::get<idx>(a));
            } else {
                m_insitu_rdata[helmholtz_slice + idx * nslices] = amrex::get<idx>(a)*dxdydz;
                m_insitu_sum_rdata[idx] += amrex::get<idx>(a)*dxdydz;
            }
        }
    );

    amrex::constexpr_for<0, m_insitu_ncp>(
        [&] (auto idx) {
            m_insitu_cdata[helmholtz_slice + idx * nslices] = amrex::get<m_insitu_nrp+idx>(a) * mid_factor;
        }
    );
}

void
Helmholtz::InSituWriteToFile (int step, amrex::Real time, int max_step, amrex::Real max_time)
{
    if (!m_use_helmholtz) return;
    if (!utils::doDiagnostics(m_insitu_period, step, max_step, time, max_time)) return;
    HIPACE_PROFILE("Helmholtz::InSituWriteToFile()");

#ifdef HIPACE_USE_OPENPMD
    // create subdirectory
    openPMD::auxiliary::create_directories(m_insitu_file_prefix);
#endif

    // zero pad the rank number;
    std::string::size_type n_zeros = 4;
    std::string rank_num = std::to_string(amrex::ParallelDescriptor::MyProc());
    std::string pad_rank_num = std::string(n_zeros-std::min(rank_num.size(), n_zeros),'0')+rank_num;

    // open file
    std::ofstream ofs{m_insitu_file_prefix + "/reduced_helmholtz." + pad_rank_num + ".txt",
        std::ofstream::out | std::ofstream::app | std::ofstream::binary};

    const int nslices_int = m_helmholtz_geom_3D.Domain().length(2);
    const std::size_t nslices = static_cast<std::size_t>(nslices_int);
    const int is_normalized_units = Hipace::m_normalized_units;

    // specify the structure of the data later available in python
    // avoid pointers to temporary objects as second argument, stack variables are ok
    const amrex::Vector<insitu_utils::DataNode> all_data{
        {"time"     , &time},
        {"step"     , &step},
        {"n_slices" , &nslices_int},
        {"z_lo"     , &m_helmholtz_geom_3D.ProbLo()[2]},
        {"z_hi"     , &m_helmholtz_geom_3D.ProbHi()[2]},
        {"is_normalized_units", &is_normalized_units},
        {"max(|a|^2)"     , &m_insitu_rdata[0], nslices},
        {"[|a|^2]"        , &m_insitu_rdata[1*nslices], nslices},
        {"[|a|^2*x]"      , &m_insitu_rdata[2*nslices], nslices},
        {"[|a|^2*x*x]"    , &m_insitu_rdata[3*nslices], nslices},
        {"[|a|^2*y]"      , &m_insitu_rdata[4*nslices], nslices},
        {"[|a|^2*y*y]"    , &m_insitu_rdata[5*nslices], nslices},
        {"axis(a)"        , &m_insitu_cdata[0], nslices},
        {"integrated", {
            {"max(|a|^2)"     , &m_insitu_sum_rdata[0]},
            {"[|a|^2]"        , &m_insitu_sum_rdata[1]},
            {"[|a|^2*x]"      , &m_insitu_sum_rdata[2]},
            {"[|a|^2*x*x]"    , &m_insitu_sum_rdata[3]},
            {"[|a|^2*y]"      , &m_insitu_sum_rdata[4]},
            {"[|a|^2*y*y]"    , &m_insitu_sum_rdata[5]}
        }}
    };

    if (ofs.tellp() == 0) {
        // write JSON header containing a NumPy structured datatype
        insitu_utils::write_header(all_data, ofs);
    }

    // write binary data according to datatype in header
    insitu_utils::write_data(all_data, ofs);

    // close file
    ofs.close();
    // assert no file errors
#ifdef HIPACE_USE_OPENPMD
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu helmholtz diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu helmholtz diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
    for (auto& x : m_insitu_cdata) x = 0.;
}
