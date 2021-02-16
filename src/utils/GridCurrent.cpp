#include "GridCurrent.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

GridCurrent::GridCurrent ()
{
    amrex::ParmParse pp("grid_current");

    if (pp.query("use_grid_current", m_use_grid_current) ) {
        pp.get("amplitude", m_amplitude);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        pp.get("position_mean", loc_array);
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
        pp.get("position_std", loc_array);
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];
    }
}

void
GridCurrent::DepositCurrentSlice (Fields& fields, const amrex::Geometry& geom, int const lev,
                                  const int islice)
{
    HIPACE_PROFILE("GridCurrent::DepositCurrentSlice()");
    using namespace amrex::literals;

    if (m_use_grid_current == 0) return;

    const auto plo = geom.ProbLoArray();
    amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();

    const amrex::GpuArray<amrex::Real, 3> pos_mean = {m_position_mean[0], m_position_mean[1],
                                                      m_position_mean[2]};
    const amrex::GpuArray<amrex::Real, 3> pos_std = {m_position_std[0], m_position_std[1],
                                                     m_position_std[2]};
    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};

    // Extract the fields currents
    amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
    amrex::MultiFab jz(S, amrex::make_alias, Comps[WhichSlice::This]["jz"], 1);

    // Extract FabArray for this box
    amrex::FArrayBox& jz_fab = jz[0];

    const amrex::Real z = plo[2] + islice*dx_arr[2];
    const amrex::Real delta_z = (z - pos_mean[2]) / pos_std[2];
    const amrex::Real long_pos_factor =  std::exp( -0.5_rt*(delta_z*delta_z) );
    const amrex::Real loc_amplitude = m_amplitude;

    for ( amrex::MFIter mfi(S, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();

        amrex::ParallelFor( bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            const amrex::Real x = plo[0] + (i+0.5_rt)*dx_arr[0];
            const amrex::Real y = plo[1] + (j+0.5_rt)*dx_arr[1];

            const amrex::Real delta_x = (x - pos_mean[0]) / pos_std[0];
            const amrex::Real delta_y = (y - pos_mean[1]) / pos_std[1];
            const amrex::Real trans_pos_factor =  std::exp( -0.5_rt*(delta_x*delta_x
                                                                    + delta_y*delta_y) );

            jz_arr(i, j, k) += loc_amplitude*trans_pos_factor*long_pos_factor;
        });
    }
}
